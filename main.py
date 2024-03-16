import sys
sys.path.insert(0, "./")
import pygad
import pyldpc
import time
from code import bin_to_sign, sign_to_bin, BER, generate_parity_check_matrix, Get_Generator_and_Parity, Code, EbN0_to_std, EbN0_to_snr
import scipy as sp
import numpy as np
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.linear import LinearEncoder, OSDecoder
from sionna.fec.utils import GaussianPriorSource
from sionna.mapping import Demapper
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.utils import BinarySource, compute_ber, BinaryCrossentropy, ebnodb2no, hard_decisions
from sionna.channel import AWGN
import tensorflow as tf
import cProfile
gpus = tf.config.list_physical_devices('GPU')

logger = tf.get_logger()

#############################################
### constants ###
N = 63 # codeword length
K = 51 # information bits per codeword
M = 1 # number of bits per symbol
BP_MAX_ITER = 10
noise_var = ebnodb2no(ebno_db=5,
                      num_bits_per_symbol=M,
                      coderate=K/N)
SNR = EbN0_to_snr(5, K/N)
# same std as in yoni's code
#############################################


def AWGN_channel(x, sigma=noise_var):
    mean = 0
    z = np.random.normal(mean, sigma, x.shape)
    y = bin_to_sign(x) + z
    return sign_to_bin(y)


# H PCM is created such that H.shape = (n-k, n)
# G generating matrix: G.shape = (k, n)
# and the following propety is satisfied: H @ G.T % 2 = 0
def test_G(G, H=None, train=False):
    if H is None:
        H = generate_parity_check_matrix(G)
    n = H.shape[1]
    k = G.shape[0]
    if train:
        t=4
        llr_source = GaussianPriorSource()
        llr = llr_source([[batch_size, n], noise_var])
    else:
        t=2
        llr_source = GaussianPriorSource()
        llr = llr_source([[batch_size, n], noise_var])
    # LDPC works llike shit decoder = LDPCBPDecoder(pcm=H, num_iter=BP_MAX_ITER if train else 1000)
    
    # OSDecoder order t
    decoder = OSDecoder(H, t=t, is_pcm=True)
    # Trying to determin batch_size given n, t


    # # # without minibatches
    x_hat = decoder(llr)
    # reconstruct b_hat - code is systematic
    b_hat = tf.slice(x_hat, [0,0], [len(llr), k])
    ber = compute_ber(tf.zeros([len(llr), k]), b_hat)


    # testing minibatches
    # ber=0
    # for i in range(int(batch_size // 100) + 1):
    #     if i < batch_size // 100:
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


def fitness_func(ga_instance, solution, solution_idx):
    # create a systematic matrix from the solution
    G = G_from_solution(solution)
    H = generate_parity_check_matrix(G)
    # check if H/G is low parity density matrix - expected 20% sparsity or less
    # sparsity_perc = np.sum(H) / np.size(H)
    # if sparsity_perc >= 0.2:
    #     # by definition of sparsity_perc, it is positive. so fitness 0 will by definition be minimal.
    #     return 0
    ber = test_G(G, H)
    fitness = -np.log(ber + 1e-20)
    # ber = test_G(G, H, train=False)
    # fitness2 = -np.log(ber + 1e-20)
    # fitness = (1/(ber + 1e-2))
    # + 1/(np.sum(solution)**2 + 10))
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


def test_crossover_func():
    num_parents_mating = 10
    offspring_size = (10, 12)
    ga_instance = type('MyClass', (), {"num_parents_mating": num_parents_mating})
    parents = np.array([[i] * offspring_size[1] for i in range(ga_instance.num_parents_mating)])
    offsprings = crossover_func(parents, offspring_size, ga_instance)
    for offspring in offsprings:
        print(offspring.reshape(3, 4).T)
        print("")


def crossover_func(parents, offspring_size, ga_instance):
    offsprings = []
    for _ in range(offspring_size[0]):
        a, b = parents[np.random.choice(ga_instance.num_parents_mating, size=2, replace=False)]
        split_points = np.random.randint(K, size=N - K)
        offspring = np.concatenate(
            [np.concatenate((a[i * K:i * K + s], b[i * K + s:(i + 1) * K])) for i, s in
             enumerate(split_points)])

        offsprings.append(offspring)

    return np.array(offsprings)


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


def test_known_matrix(code_type="LDPC", n=49, k=24):
    G, H = Get_Generator_and_Parity(Code(code_type, n, k))
    ber = test_G(G, H, train=False)
    print(f"BER: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")
    ber = test_G(G, H, train=True)
    print(f"BER: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")

def test_known_matrix_2(code_type="LDPC", n=49, k=24):
    G, H = Get_Generator_and_Parity(Code(code_type, n, k))
    ber = test_G(G, H, train=False)
    print(f"BER: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")
    ber = test_G(G, H, train=True)
    print(f"BER: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")


def G_from_solution(solution):
    return np.hstack((np.eye(K), solution.reshape(N - K, K).T))

if __name__ == '__main__':
    # each matrix is a systematic matrix such that matrix.reshape((k, n)) is the systematic matrix
    start = time.time()
    num_initial_population = 200
    batch_size = 500
    num_generations=500
    function_inputs = np.random.randint(
        2, size=(K, batch_size))
    initial_population = np.random.choice(
        a=[0, 1], p=[0.8, 0.2], size=(num_initial_population, K*(N-K)))
    channel_func = AWGN_channel
    # test_known_matrix(code_type="LDPC", n=49, k=24)
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=50,
                           sol_per_pop=20,
                           initial_population=initial_population,
                           gene_type=np.uint8,
                           fitness_func=fitness_func,
                           # crossover_type=crossover_func,
                           mutation_type=mutation_func,
                           on_generation=on_generation,
                           crossover_type=crossover_func,
                           # parallel_processing=['process', 2],
                           parallel_processing=['thread', 16]
                           )

    # Running the GA to optimize the parameters of the function.
    with cProfile.Profile() as pr:
        with tf.xla.experimental.jit_scope():
            ga_instance.run()
        pr.dump_stats(f"./profiler_stats_for_n_{N}_k_{K}_initpop_{num_initial_population}_gencount_{num_generations}_bsize_{batch_size}_pmating_50_sol_per_pop_20")
    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    G = G_from_solution(solution)
    ber = test_G(G, train=False)
    print(f"Parameters of the best solution : {solution}")
    print(f"Generator matrix of best solution:\n{G}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"BER value of best solution: {ber}")
    print(f"Index of the best solution : {solution_idx}")
    # print(f"{ np.sum(solution*function_inputs)}")
    print(f"Running Time: {time.time() - start} [s]")
    ga_instance.plot_fitness(savedir="./")
    ga_instance.save(f"best_sol_for_n_{N}_k_{K}_initpop_{num_initial_population}_gencount_{num_generations}")
