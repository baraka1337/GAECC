import pygad
import pyldpc
import time
from code import bin_to_sign, BER, generate_parity_check_matrix, Get_Generator_and_Parity, Code, EbN0_to_std
import numpy as np
#############################################
### constants ###
N = 49
K = 24
BP_MAX_ITER = 10
BP_SNR = 3  # [db]
#SIGMA = (10 ** (-BP_SNR / 10))**0.5
SIGMA = EbN0_to_std(5, K/N)
# same std as in yoni's code
# SIGMA = EbN0_to_std(7, K/N)
# BP_SNR = -10 *BP_SNR math.log(SIGMA**2, 10)

# function_inputs = np.random.uniform(-5, 5, size=sample_size)
#############################################


def AWGN_channel(x, sigma=SIGMA):
    mean = 0
    z = np.random.normal(mean, sigma, x.shape)
    y = bin_to_sign(x) + z
    return y


# H PCM is created such that H.shape = (n-k, n)
# and the following propety is satisfied: H @ G.T % 2 = 0


def test_G(G, H=None, train=True):
    if H is None:
        H = generate_parity_check_matrix(G)
    if train:
        x_vec = np.zeros((function_inputs.shape[1], G.shape[1]))
    else:
        x_vec = G.T @ function_inputs % 2
        x_vec = x_vec.T
    y_vec = AWGN_channel(x_vec)

    x_pred_vec = pyldpc.decode(H, y_vec.T, BP_SNR, BP_MAX_ITER if train else 1000)
    x_pred_vec = x_pred_vec.T
    return BER(x_vec, x_pred_vec)


def fitness_func(ga_instance, solution, solution_idx):
    # create a systematic matrix from the solution
    G = np.hstack((np.eye(K), solution.reshape(K, N-K)))
    H = generate_parity_check_matrix(G)
    # check if H/G is low parity density matrix - expected 20% sparsity or less
    sparsity_perc = np.sum(H) / np.size(H)
    if sparsity_perc >= 0.2:
        # by definition of sparsity_perc, it is positive. so fitness 0 will by definition be minimal.
        return 0
    ber = test_G(G, H)
    fitness = 1/(ber + 1e-2) + \
        1/(np.sum(solution)**2 + 10)
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
    sample_size = 1000
    function_inputs = np.random.randint(
        2, size=(K, sample_size)).astype(bool)
    initial_population = np.random.choice(
        a=[0, 1], p=[0.8, 0.2], size=(num_initial_population, K*(N-K)))
    channel_func = AWGN_channel
    G, H = Get_Generator_and_Parity(Code("LDPC", 49, 24))
    ber = test_G(G.astype(bool), H, train=False)
    print(f"BER: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")
    ber = test_G(G.astype(bool), H, train=True)
    print(f"BER: {ber}")
    print(f"Negative natural logarithm of Bit Error Rate: {-np.log(ber)}")
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
            f"BER value of best solution: {(solution_fitness - 1/(np.sum(solution)**2 + 10))**-1 - 1e-2}")
        print(f"Index of the best solution : {solution_idx}")
        # print(f"{ np.sum(solution*function_inputs)}")
        print(f"Running Time: {time.time()-start} [s]")
        ga_instance.plot_fitness()
        ga_instance.save("best_sol")
