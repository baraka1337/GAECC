import pygad
import numpy
import torch
import pyldpc
import time
import math


def set_seed(seed=42):
    numpy.random.seed(seed)


def sign_to_bin(x):
    return 0.5 * (1 - x)


def bin_to_sign(x):
    return 1 - 2 * x


def EbN0_to_std(EbN0, rate):
    snr = EbN0 + 10. * numpy.log10(2 * rate)
    return numpy.sqrt(1. / (10. ** (snr / 10.)))


def BER(x_pred, x_gt):
    return numpy.mean((x_pred != x_gt)).item()


def FER(x_pred, x_gt):
    return numpy.mean(numpy.any(x_pred != x_gt, dim=1).float()).item()


#############################################
### constants ###
N = 63
K = 51
BP_MAX_ITER = 10
BP_SNR = 3  # [db]
SIGMA = (10 ** (-BP_SNR / 10))**0.5
# same std as in yoni's code
# SIGMA = EbN0_to_std(7, K/N)
# BP_SNR = -10 *BP_SNR math.log(SIGMA**2, 10)
num_initial_population = 1000
sample_size = 1000
# function_inputs = numpy.random.uniform(-5, 5, size=sample_size)
#############################################


def AWGN_channel(x, sigma=SIGMA):
    shape = (sample_size, N)
    mean = 0
    z = numpy.random.normal(mean, sigma, shape)
    y = bin_to_sign(x) + z
    return y


# H PCM is created such that H.shape = (n-k, n)
# and the following propety is satisfied: H @ G.T % 2 = 0
def generate_parity_check_matrix(G):
    k, n = G.shape
    H = numpy.hstack((G[:, k:].T, numpy.eye(n - k)))
    return H


def fitness_func(ga_instance, solution, solution_idx):
    # create a systematic matrix from the solution
    G = numpy.hstack((numpy.eye(K), solution.reshape(K, N-K)))
    H = generate_parity_check_matrix(G)
    # check if H/G is low parity density matrix - expected 20% sparsity or less
    sparsity_perc = numpy.sum(H) / numpy.size(H)
    if sparsity_perc >= 0.2:
        # by definition of sparsity_perc, it is positive. so fitness 0 will by definition be minimal.
        return 0
    x_vec = G.T @ function_inputs % 2
    x_vec = x_vec.T
    y_vec = AWGN_channel(x_vec)
    # decoded codewords
    x_pred_vec = numpy.zeros(y_vec.shape)
    # for i in range(y_vec.shape[1]):
    #     try:
    #        x_pred_vec[i] = pyldpc.decode(H, y_vec[i], BP_SNR, BP_MAX_ITER)
    #     except Exception as e:
    #         print(e)
    #         import ipdb; ipdb.set_trace()
    """
    decoding implementation with pyldpc
    """
    try:
        x_pred_vec = pyldpc.decode(H, y_vec.T, BP_SNR, BP_MAX_ITER)
        x_pred_vec = x_pred_vec.T
    except Exception as e:
        print(e)
        import ipdb
        ipdb.set_trace()

    """
    decoding implementation with non-gaussian channels - Not parrallel computation for now
        model = AWGN_channel
    tg = TannerGraph.from_biadjacency_matrix(H, channel_model=model)
    bp = BeliefPropagation(tg, H, max_iter=10)
    for i in range(y_vec.shape[1]):
        try:
           x_pred_vec[i] = bp.decode(y_vec[i])
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
    """
    fitness = 1/(BER(x_vec, x_pred_vec) + 1e-2) + \
        1/(numpy.sum(solution)**2 + 10)
    return fitness


def mutation_func(offspring, ga_instance):
    # This is random mutation that mutates a single gene.
    for chromosome_idx in range(offspring.shape[0]):
        # Make some random changes in 1 or more genes.
        # trying to flip 10 bits and see how it affects
        for i in range(10):
            random_gene_idx = numpy.random.choice(range(offspring.shape[1]))
            offspring[chromosome_idx, random_gene_idx] ^= 1

    return offspring


# def crossover_func(parents, offspring_size, ga_instance):
#    import ipdb;ipdb.set_trace()
last_fitness = 0
start_g = time.time()


def on_generation(ga_instance):
    global last_fitness
    global start_g
    print(f"Generation = {ga_instance.generations_completed}")
    solution, solution_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[:2]
    print(
        f"Fitness    = {solution_fitness}")
    print(
        f"Change in Fitness     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    print(
        f"BER value of best solution: {(solution_fitness - 1/(numpy.sum(solution)**2 + 10))**-1 - 1e-2}")
    end_g = time.time()
    print(f"Generation Running Time: {end_g-start_g} [s]")
    start_g = end_g
    last_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[1]


if __name__ == '__main__':
    # each matrix is a systematic matrix such that matrix.reshape((k, n)) is the systematic matrix
    start = time.time()
    function_inputs = numpy.random.randint(2, size=(K, sample_size))
    initial_population = numpy.random.choice(
        a=[0, 1], p=[0.8, 0.2], size=(num_initial_population, K*(N-K)))
    channel_func = AWGN_channel
    ga_instance = pygad.GA(num_generations=30,
                           num_parents_mating=50,
                           sol_per_pop=20,
                           initial_population=initial_population,
                           gene_type=numpy.uint8,
                           fitness_func=fitness_func,
                           # crossover_type=crossover_func,
                           mutation_type=mutation_func,
                           on_generation=on_generation,
                           # parallel_processing=['process', 2],
                           parallel_processing=['thread', 4]
                           )

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(
        f"Generator matrix of best solution: {numpy.hstack((numpy.eye(K), solution.reshape(K, N-K)))}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(
        f"BER value of best solution: {(solution_fitness - 1/(numpy.sum(solution)**2 + 10))**-1 - 1e-2}")
    print(f"Index of the best solution : {solution_idx}")
    # print(f"{ numpy.sum(solution*function_inputs)}")
    print(f"Running Time: {time.time()-start} [s]")
    ga_instance.plot_fitness()
    ga_instance.save("best_sol")
