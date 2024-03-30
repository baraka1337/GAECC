import sys
import os
sys.path.insert(0, "./")
os.environ["TF_ENABLE_XLA"] = '1'
os.environ["TF_XLA_ENABLE_XLA_DEVICES"] = '1'
os.environ["TF_XLA_FLAGS"] = '--tf_xla_auto_jit=1, --tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tensorflow as tf
import pygad
import pyldpc
import time
from code import bin_to_sign, sign_to_bin, BER, generate_parity_check_matrix, Get_Generator_and_Parity, Code, EbN0_to_std, EbN0_to_snr
import scipy as sp
import numpy as np
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.linear import OSDecoder
from sionna.fec.utils import GaussianPriorSource, gm2pcm
from sionna.utils import compute_ber, ebnodb2no, BinarySource
from sionna.channel import AWGN, BinarySymmetricChannel

import cProfile
from concurrent.futures import ThreadPoolExecutor
from copy import copy

from matplotlib import pyplot as plt
from tqdm import tqdm

def config_tf():
    tf.config.optimizer.set_jit(True)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # tf.config.set_logical_device_configuration(
                #     gpu,
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=30000)])
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

# logger = tf.get_logger()

#############################################
### constants ###
# N = 63 # codeword length
# K = 51 # information bits per codeword
# M = 1 # number of bits per symbol
# noise_var = ebnodb2no(ebno_db=5,
#                       num_bits_per_symbol=M,
#                       coderate=K/N)
# SNR = EbN0_to_snr(5, K/N)
# same std as in yoni's code
#############################################
BP_SNR = 3 # [db]
BP_MAX_ITER = 5

def AWGN_channel_calc_LLR(y, sigma):
    return 2*y/(sigma**2)

def apply_function_with_threads(func, iterator, *args, **kwargs):
    """
    Apply a function with threads to each item in an iterator using ThreadPoolExecutor.

    Args:
    - iterator: Iterator containing items to apply the function on.
    - func: Function to be applied on each item.
    - *args: Extra positional arguments to be passed to the function.
    - **kwargs: Extra keyword arguments to be passed to the function.
    """
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Map each item in the iterator to the function with given arguments
        futures = {executor.submit(func, *item, *args, **kwargs): item for item in iterator}

        # Wait for all threads to finish
        for future in tqdm(futures):
            future.result()


class GA:
    def __init__(
            self,
            k,
            n,
            num_initial_population,
            sample_size,
            num_parents_mating,
            offspring_size,
            p_mutation=0.05,
            num_generations=5,
            ebn0=5,
    ):
        self.k = k
        self.n = n
        self.num_initial_population = num_initial_population
        self.sample_size = sample_size
        self.num_parents_mating = num_parents_mating
        initial_population_size = (self.num_initial_population, self.k, (self.n - self.k))
        self.population = tf.Variable(tf.convert_to_tensor(np.random.choice(a=[0,1], size=initial_population_size), dtype=tf.float32))
        self.channel_func = AWGN_channel
        self.fitness_result = np.zeros(num_initial_population)
        self.nlog = np.zeros(num_initial_population)
        self.offspring_size = offspring_size
        self.p_mutation = p_mutation
        self.num_generations = num_generations

        self.fitness_result_normalize = None
        self.start_generation_time = 0
        self.last_fitness = 0
        self.best_solution = None
        self.best_solution_fitness = 0
        self.sigma = EbN0_to_std(ebn0, self.k / self.n)
        self.bests_nlog = []

    def fitness_single(self, solution, i, gamma=3):
        # create a systematic matrix from the solution
        delta = 1e-20
        G = G_from_solution(solution, self.k)
        H = generate_parity_check_matrix(G)
        ber = test_G(G, self.sample_size, self.sigma, H)
        fitness = (1 / (ber + delta)) ** gamma
        self.fitness_result[i] = fitness
        self.nlog[i] = -np.log(ber + delta)

    def fitness(self):
        with tf.profiler.experimental.Trace(f"Fitness"):
            # apply_function_with_threads(self.fitness_single, enumerate(self.population))
            dataset = tf.data.Dataset.from_tensor_slices(self.population)

            # Enumerate the dataset
            enumerated_dataset = dataset.enumerate()  # start specifies the offset for the enumeration

            for i, solution in tqdm(enumerated_dataset, total=self.num_initial_population):
                if self.fitness_result[i] == 0:
                    solution = tf.cast(solution, dtype=tf.int32).numpy()
                    self.fitness_single(solution, i)
                    tf.keras.backend.clear_session()
            self.fitness_result_normalize = self.fitness_result / np.sum(self.fitness_result)
            best_solution_index = np.argmax(self.fitness_result)
            self.best_solution = copy(self.population[best_solution_index])
            self.best_solution_fitness = self.fitness_result[best_solution_index]
            self.bests_nlog.append(self.nlog[best_solution_index])

    def crossover(self):
        # Selecting parents based on the normalized fitness results
        indices_of_parents_mating = tf.random.categorical(tf.math.log([self.fitness_result_normalize]), self.num_parents_mating, dtype=tf.int32)[0]
        
        # Select random indices for crossover
        indices = tf.random.uniform(shape=(2, self.offspring_size), minval=0, maxval=self.num_parents_mating, dtype=tf.int32)
        
        # Gather selected parents
        selected_parents = tf.gather(self.population, indices_of_parents_mating)
        a = tf.gather(selected_parents, indices[0], axis=0)
        b = tf.gather(selected_parents, indices[1], axis=0)

        # Generate random crossover points
        random_points = tf.random.uniform(shape=(self.offspring_size, self.n - self.k), minval=0, maxval=self.k, dtype=tf.int32)
        
        # Create mask for crossover
        mask = tf.range(self.k) < random_points[..., tf.newaxis]
        mask = tf.transpose(mask, perm=[0, 2, 1])
        
        # Apply crossover mask
        offsprings = tf.where(mask, a, b)

        return offsprings

    def mutation(self, offsprings):
        # Assuming offsprings is a TensorFlow tensor
        offsprings_shape = tf.shape(offsprings)
        mutation_mask = tf.random.uniform(offsprings_shape) < self.p_mutation
        
        # XOR operation is not directly supported between a boolean mask and integer tensors in TensorFlow,
        # so we convert the boolean mask to the same dtype as offsprings and perform bitwise XOR where mutation_mask is True.
        mutation_values = tf.cast(mutation_mask, dtype=tf.int32) * tf.constant(1, dtype=tf.int32)
        offsprings = tf.bitwise.bitwise_xor(tf.cast(offsprings, dtype=tf.int32), mutation_values)

        return offsprings

    def run(self):
        plt.figure()
        self.start_generation_time = time.time()
        for i in tf.range(self.num_generations):
            self.fitness()
            offsprings = self.crossover()
            offsprings = tf.cast(self.mutation(offsprings), dtype=tf.float32)
            
            # Using TensorFlow operations
            argsort_result = tf.argsort(self.fitness_result, direction='ASCENDING')
            offsprings_indices = argsort_result[:self.offspring_size]

            def update_population(offspring, index):
                self.population[index].assign(tf.cast(offspring, dtype=tf.float32))
                return self.population
            # Updating population with new offsprings
            tf.map_fn(lambda x: update_population(offsprings[x], offsprings_indices[x]), tf.range(self.offspring_size), dtype=tf.float32)
            
            self.end_generation(i)
            
            # Resetting fitness results of the updated population members
            # tf.scatter_update(self.fitness_result, offsprings_indices, tf.zeros(self.offspring_size))
            self.fitness_result[offsprings_indices.numpy()] = np.zeros(self.offspring_size)
        self.end()

    def end_generation(self, generations_index):
        change_in_fitness = self.best_solution_fitness - self.last_fitness
        # G = G_from_solution(self.best_solution, self.k)
        # ber = test_G(G, self.sample_size, self.sigma)
        nlog = self.bests_nlog[-1]
        ber = np.exp(-nlog)
        print(f"Generation = {generations_index}")
        print(f"Fitness    = {self.best_solution_fitness}")
        print(f"Change in Fitness     = {change_in_fitness}")
        print(f"BER value of best solution: {ber}")
        print(f"Negative natural logarithm of Bit Error Rate: {nlog}")
        end_generation_time = time.time()
        print(f"Generation Running Time: {end_generation_time - self.start_generation_time} [s]")
        self.start_generation_time = end_generation_time
        self.last_fitness = self.best_solution_fitness
        plt.plot(sorted(self.nlog), label=f"gen {generations_index}")
        plt.savefig(os.path.join(RESULTS_DIR, f"generation_{generations_index}.png"))

    def end(self):
        plt.grid()
        plt.legend()
        plt.title("Best Negative natural logarithm of Bit Error Rate")
        plt.xlabel("Generation")
        plt.ylabel("-log(ber)")
        # plt.show()
        plt.savefig(os.path.join(RESULTS_DIR, "1.png"))
        plt.figure()
        plt.plot(self.bests_nlog)
        plt.title("Best Negative natural logarithm of Bit Error Rate")
        plt.xlabel("Generation")
        plt.ylabel("-log(ber)")
        plt.grid()
        # plt.show()
        plt.savefig(os.path.join(RESULTS_DIR, "2.png"))


def AWGN_channel(x, sigma):
    mean = 0
    z = np.random.normal(mean, sigma, x.shape)
    y = bin_to_sign(x) + z
    return y


# H PCM is created such that H.shape = (n-k, n)
# G generating matrix: G.shape = (k, n)
# and the following propety is satisfied: H @ G.T % 2 = 0
def test_G_sionna_os_decoder(G, sample_size, sigma, H=None, train=True):
    noise_var = sigma**2
    if H is None:
        H = generate_parity_check_matrix(G)
    n = H.shape[1]
    k = G.shape[0]
    # llr_source = GaussianPriorSource()
    # llr = llr_source([[sample_size, n], noise_var])
    x_vec = np.zeros((sample_size, G.shape[1]))
    # AWGN channel
    y_vec = AWGN_channel(x_vec, sigma)
    # minus because decoder expects log(p(x=1)/p(x=0)) and the AWGN_channel_calc_LLR function
    # calculates log(p(x=0)/p(x=1))
    llr = -AWGN_channel_calc_LLR(y_vec, sigma)

    # Binary Symmetric Channel
    # bsc = BinarySymmetricChannel(return_llrs=True)
    # llr = bsc((x_vec, 0.1))
    if train:
        t=2
    else:
        t=2
    # OSDecoder order t
    decoder = OSDecoder(H, t=t, is_pcm=True)
    # Trying to determin sample_size given n, t


    # # # without minibatches
    x_hat = decoder(llr)
    # reconstruct b_hat - code is systematic
    b_hat = tf.slice(x_hat, [0,0], [len(llr), k])
    return compute_ber(x_vec, x_hat)



# H PCM is created such that H.shape = (n-k, n)
# G generating matrix: G.shape = (k, n)
# and the following propety is satisfied: H @ G.T % 2 = 0
def test_G_sionna_ldpc_decoder(G, sample_size, sigma, H=None, train=True):
    if H is None:
        H = generate_parity_check_matrix(G)
    n = H.shape[1]
    k = G.shape[0]
    
    if train: # training is on all zero codeword
        x_vec = np.zeros((n, sample_size))
    else: #  generating actual data
        bs = BinarySource()
        data_vec = bs((sample_size, k))
        x_vec = G.T @  tf.transpose(data_vec) % 2
    # AWGN channel
    y_vec = AWGN_channel(x_vec, sigma)
    # minus because decoder expects log(p(x=1)/p(x=0)) and the AWGN_channel_calc_LLR function
    # calculates log(p(x=0)/p(x=1))
    llr = tf.transpose(-AWGN_channel_calc_LLR(y_vec, sigma))

    # Binary Symmetric Channel
    # bsc = BinarySymmetricChannel(return_llrs=True)
    # llr = bsc((x_vec, 0.1))
    # if train:
    #     t=2
    # else:
    #     t=2
    # H = H>=1
    # H = sp.sparse.csc_matrix(H)
    decoder = LDPCBPDecoder(pcm=H, num_iter=BP_MAX_ITER  if train else 1000)

    # # # without minibatches
    x_hat = decoder(llr)

    # reconstruct b_hat - code is systematic
    b_hat = tf.slice(x_hat, [0,0], [len(llr), k])

    return compute_ber(tf.cast(tf.transpose(x_vec), dtype=tf.float32), x_hat)

def test_G(G, sample_size, sigma, H=None, train=True, simulate_G_func=test_G_sionna_ldpc_decoder):
    if simulate_G_func is not None:
        return simulate_G_func(G, sample_size, sigma, H, train)
    if H is None:
        H = generate_parity_check_matrix(G)
    
    x_vec = np.zeros((sample_size, G.shape[1]))
    #
    # if train:
    #     x_vec = np.zeros((int(function_inputs.shape[1] // 10), G.shape[1]))
    # else:
    #     x_vec = G.T @ function_inputs % 2
    #     x_vec = x_vec.T
    y_vec = AWGN_channel(x_vec, sigma)
    # var = 10 ** (-snr / 10)
    snr = -np.math.log(sigma**2, 10) * 10
    x_pred_vec = pyldpc.decode(H, y_vec.T, snr, BP_MAX_ITER if train else 1000)
    x_pred_vec = x_pred_vec.T
    return BER(x_vec, x_pred_vec)




def G_from_solution(solution, k):
    return np.hstack((np.eye(k), solution))


if __name__ == '__main__':
    config_tf()
    num_initial_population = 700
    RESULTS_DIR = "./results_tf_main_sionna_decoder"
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                   python_tracer_level = 1,
                                                   device_tracer_level = 1)
    tf.profiler.experimental.start(os.path.join(RESULTS_DIR, 'tensorboard_logs'), options)  # Logs will be saved in the 'logs' directory
    
    ga = GA(
        k=24,
        n=49,
        num_initial_population=num_initial_population,
        sample_size=1000,
        num_parents_mating=int(num_initial_population * 0.2),
        offspring_size=int(num_initial_population * 0.8),
        p_mutation=0.005,
        num_generations=50
    )

    # Running the GA to optimize the parameters of the function.
    ga.run()
        # pr.dump_stats(os.path.join(RESULTS_DIR, f"./profiler_stats_for_n_{ga.n}_k_{ga.k}_initpop_{num_initial_population}_gencount_{ga.num_generations}_bsize_{ga.sample_size}_pmating_{ga.num_parents_mating}"))
    tf.profiler.experimental.stop()
    # Returning the details of the best solution.`
