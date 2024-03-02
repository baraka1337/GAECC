import pygad
import pyldpc
import time
from code import bin_to_sign, sign_to_bin, BER, generate_parity_check_matrix, Get_Generator_and_Parity, Code, EbN0_to_std, EbN0_to_snr
import numpy as np
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.linear import LinearEncoder
from sionna.fec.utils import GaussianPriorSource
from sionna.mapping import Demapper
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.utils import BinarySource, compute_ber, BinaryCrossentropy, ebnodb2no, hard_decisions
from sionna.channel import AWGN
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

#############################################
### constants ###
batch_size = 1024
N = 49 # codeword length
K = 24 # information bits per codeword
M = 1 # number of bits per symbol
BP_MAX_ITER = 10
noise_var = ebnodb2no(ebno_db=5,
                      num_bits_per_symbol=M,
                      coderate=K/N)
SNR = EbN0_to_snr(5, K/N)
G_MAT_SPARSITY_PRECENTAGE_UPPER_BOUND = 0.2
# same std as in yoni's code
#############################################


def AWGN_channel(x, sigma=noise_var):
    mean = 0
    z = np.random.normal(mean, sigma, x.shape)
    y = bin_to_sign(x) + z
    return sign_to_bin(y)


# H PCM is created such that H.shape = (n-k, n)
# and the following propety is satisfied: H @ G.T % 2 = 0


def test_G(G, H=None, train=False):
    if H is None:
        H = generate_parity_check_matrix(G)
    if train:
        llr_source = GaussianPriorSource()
        llr = llr_source([[batch_size, N], noise_var])

    else:
        llr_source = GaussianPriorSource()
        llr = llr_source([[batch_size, N], noise_var])
    decoder = LDPCBPDecoder(pcm=H, num_iter=BP_MAX_ITER if train else 1000)

    # x_pred_vec = pyldpc.decode(H, y_vec.T, BP_SNR, BP_MAX_ITER if train else 1000)
    x_hat = decoder(llr)
    # reconstruct b_hat - code is systematic
    b_hat = tf.slice(x_hat, [0,0], [batch_size, K])

    ber = compute_ber(tf.zeros([batch_size, K]), b_hat)
    return ber.numpy()


def fitness_func(ga_instance, solution, solution_idx):
    # create a systematic matrix from the solution
    G = np.hstack((np.eye(K), solution.reshape(K, N-K)))
    H = generate_parity_check_matrix(G)
    # check if H/G is low parity density matrix - expected 20% sparsity or less
    sparsity_perc = np.sum(H) / np.size(H)
    if sparsity_perc >= G_MAT_SPARSITY_PRECENTAGE_UPPER_BOUND:
        # by definition of sparsity_perc, it is positive. so fitness 0 will by definition be minimal.
        return 0
    ber = test_G(G, H)
    fitness = 1/(ber + 1e-2)
    return fitness


def mutation_func(offspring, ga_instance):
    # This is random mutation that mutates a single gene.
    for chromosome_idx in range(offspring.shape[0]):
        # Make some random changes in 1 or more genes.
        # trying to flip 10 bits and see how it affects
        for i in range(10):
            random_gene_idx = np.random.choice(range(offspring.shape[1]))
            offspring[chromosome_idx, random_gene_idx] ^= 1

    return offspring


last_fitness = 0
start_g = time.time()


def on_generation(ga_instance):
    global last_fitness
    global start_g
    change_in_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
    solution, solution_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[:2]
    G = np.hstack((np.eye(K), solution.reshape(K, N-K)))
    ber = test_G(G, train=False)
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {solution_fitness}")
    print(f"Change in Fitness     = {change_in_fitness}")
    print(f"BER value of best solution: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")
    end_g = time.time()
    print(f"Generation Running Time: {end_g-start_g} [s]")
    start_g = end_g
    last_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[1]


if __name__ == '__main__':
    # each matrix is a systematic matrix such that matrix.reshape((k, n)) is the systematic matrix
    start = time.time()
    num_initial_population = 100
    initial_population = np.random.choice(
        a=[0, 1], p=[0.8, 0.2], size=(num_initial_population, K*(N-K)))
    G, H = Get_Generator_and_Parity(Code("LDPC", 49, 24))
    ber = test_G(G, H, train=False)
    print(f"BER: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")
    # ber = test_G(G.astype(bool), H, train=True)
    # print(f"BER: {ber}")
    # print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")
    # import ipdb
    # ipdb.set_trace()
    ga_instance = pygad.GA(num_generations=30,
                           num_parents_mating=50,
                           sol_per_pop=20,
                           initial_population=initial_population,
                           gene_type=np.uint8,
                           fitness_func=fitness_func,
                           # crossover_type=crossover_func,
                           mutation_type=mutation_func,
                           on_generation=on_generation,
                           # parallel_processing=['process', 2],
                           parallel_processing=['thread', 4]
                           )

    # Running the GA to optimize the parameters of the function.
    try:
        ga_instance.run()
    except:
        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(
            ga_instance.last_generation_fitness)
        print(f"Parameters of the best solution : {solution}")
        print(
            f"Generator matrix of best solution: {np.hstack((np.eye(K), solution.reshape(K, N-K)))}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(
            f"BER value of best solution: {(solution_fitness)**-1 - 1e-2}")
        print(f"Index of the best solution : {solution_idx}")
        # print(f"{ np.sum(solution*function_inputs)}")
        print(f"Running Time: {time.time()-start} [s]")
        ga_instance.plot_fitness(savedir="~/GAECC/")
        ga_instance.save("~/GAECC/best_sol")
