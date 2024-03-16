import sys
import os
# os.system("python3.8 -m pip install tqdm")
sys.path.insert(0, "./")
import pygad
import pyldpc
import time
from code import bin_to_sign, sign_to_bin, BER, generate_parity_check_matrix, Get_Generator_and_Parity, Code, EbN0_to_std, EbN0_to_snr
import scipy as sp
import numpy as np
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.linear import OSDecoder
from sionna.fec.utils import GaussianPriorSource
from sionna.utils import compute_ber, ebnodb2no
from sionna.channel import AWGN
import tensorflow as tf
import cProfile
from concurrent.futures import ThreadPoolExecutor
from copy import copy

from matplotlib import pyplot as plt
from tqdm import tqdm
gpus = tf.config.list_physical_devices('GPU')
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
BP_SNR = 5
BP_MAX_ITER = 5


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
            ebn0=1,
    ):
        self.k = k
        self.n = n
        self.num_initial_population = num_initial_population
        self.sample_size = sample_size
        self.num_parents_mating = num_parents_mating
        initial_population_size = (self.num_initial_population, self.k, (self.n - self.k))
        self.population = np.random.choice(a=[0, 1], size=initial_population_size)
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
        # apply_function_with_threads(self.fitness_single, enumerate(self.population))
        for i, solution in tqdm(enumerate(self.population), total=self.num_initial_population):
            if self.fitness_result[i] == 0:
                self.fitness_single(solution, i)
        self.fitness_result_normalize = self.fitness_result / np.sum(self.fitness_result)
        best_solution_index = np.argmax(self.fitness_result)
        self.best_solution = copy(self.population[best_solution_index])
        self.best_solution_fitness = self.fitness_result[best_solution_index]
        self.bests_nlog.append(self.nlog[best_solution_index])

    def crossover(self):
        indices_of_parents_mating = np.random.choice(self.num_initial_population, size=self.num_parents_mating,
                                                     p=self.fitness_result_normalize, replace=False)
        # indices = np.random.choice(self.num_parents_mating, size=(2, self.offspring_size))
        selected_parents_normalized_p = self.fitness_result_normalize[indices_of_parents_mating] / np.sum(self.fitness_result_normalize[indices_of_parents_mating])
        indices = np.random.choice(self.num_parents_mating, size=(2, self.offspring_size), p=selected_parents_normalized_p)
        a = self.population[indices_of_parents_mating][indices[0]]
        b = self.population[indices_of_parents_mating][indices[1]]

        random_points = np.random.randint(self.k, size=(self.offspring_size, self.n - self.k))
        mask = np.arange(self.k) < random_points[:, :, None]
        mask = mask.transpose((0, 2, 1))
        offsprings = np.where(mask, a, b)

        return offsprings

    def mutation(self, offsprings):
        mutation_mask = np.random.rand(*offsprings.shape) < self.p_mutation
        offsprings[mutation_mask] ^= 1

    def run(self):
        plt.figure()
        self.start_generation_time = time.time()
        for i in range(self.num_generations):
            self.fitness()
            offsprings = self.crossover()
            self.mutation(offsprings)
            argsort_result = np.argsort(self.fitness_result)
            offsprings_indices = argsort_result[:self.offspring_size]
            self.population[offsprings_indices] = offsprings
            self.end_generation(i)
            self.fitness_result[offsprings_indices] = np.zeros(self.offspring_size)
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
    return sign_to_bin(y)


# H PCM is created such that H.shape = (n-k, n)
# G generating matrix: G.shape = (k, n)
# and the following propety is satisfied: H @ G.T % 2 = 0
def test_G_sionna_os_decoder(G, sample_size, sigma, H=None, train=True):
    noise_var = sigma
    if H is None:
        H = generate_parity_check_matrix(G)
    n = H.shape[1]
    k = G.shape[0]
    llr_source = GaussianPriorSource()
    llr = llr_source([[sample_size, n], noise_var])
    if train:
        t=2
    else:
        t=2
    # LDPC works llike shit decoder = LDPCBPDecoder(pcm=H, num_iter=BP_MAX_ITER if train else 1000)
    # OSDecoder order t
    decoder = OSDecoder(H, t=t, is_pcm=True)
    # Trying to determin sample_size given n, t


    # # # without minibatches
    x_hat = decoder(llr)
    # reconstruct b_hat - code is systematic
    b_hat = tf.slice(x_hat, [0,0], [len(llr), k])
    ber = compute_ber(tf.zeros([len(llr), k]), b_hat)


    # testing minibatches
    # ber=0
    # for i in range(int(sample_size // 100) + 1):
    #     if i < sample_size // 100:
    #         mini_batch = llr[100 * i:100 * (i+1),:]
    #         # x_pred_vec = pyldpc.decode(H, y_vec.T, BP_SNR, BP_MAX_ITER if train else 1000)
    #     else:
    #         mini_batch = llr[100 * i:, :]
    #     x_hat = decoder(mini_batch)
    #     # reconstruct b_hat - code is systematic
    #     b_hat = tf.slice(x_hat, [0,0], [len(mini_batch), G.shape[0]])
    #     ber += compute_ber(tf.zeros([len(mini_batch), G.shape[0]]), b_hat)
    # print(ber)
    return ber.numpy()

def test_G(G, sample_size, sigma, H=None, train=True, simulate_G_func=test_G_sionna_os_decoder):
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

    x_pred_vec = pyldpc.decode(H, y_vec.T, BP_SNR, BP_MAX_ITER if train else 1000)
    x_pred_vec = x_pred_vec.T
    return BER(x_vec, x_pred_vec)


def G_from_solution(solution, k):
    return np.hstack((np.eye(k), solution))

if __name__ == '__main__':
    num_initial_population = 840
    RESULTS_DIR = "./results_2"
    ga = GA(
        k=24,
        n=49,
        num_initial_population=num_initial_population,
        sample_size=1000,
        num_parents_mating=int(num_initial_population * 0.2),
        offspring_size=int(num_initial_population * 0.8),
        p_mutation=0.1,
        num_generations=200
    )

    # Running the GA to optimize the parameters of the function.
    with cProfile.Profile() as pr:
        ga.run()
        pr.dump_stats(os.path.join(RESULTS_DIR, f"./profiler_stats_for_n_{ga.n}_k_{ga.k}_initpop_{num_initial_population}_gencount_{ga.num_generations}_bsize_{ga.sample_size}_pmating_{ga.num_parents_mating}"))
    # Returning the details of the best solution.