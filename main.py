from code import AWGN_channel, G_from_solution, generate_parity_check_matrix, EbN0_to_std, EbN0_to_snr, test_G, DEFAULT_BP_MAX_ITER  # type: ignore
from matplotlib import pyplot as plt
import sys

from utils import apply_function_with_processes

sys.path.insert(0, "./")
import numpy as np
import pickle
import time


def fitness_single(
    solution, last_fitness_result, k, sample_size, sigma, snr, delta, gamma, bp_iter
):
    """
    Calculates the fitness value for a single solution in a genetic algorithm.

    Parameters:
    - solution: The solution for which to calculate the fitness value.
    - last_fitness_result: The fitness value calculated in the previous iteration. If non-zero, it is returned as is.
    - k: The number of bits in the solution.
    - sample_size: The number of samples to use for testing.
    - sigma: The standard deviation of the noise.
    - snr: The signal-to-noise ratio.
    - delta: A small constant to avoid division by zero.
    - gamma: The exponent used in the fitness calculation.
    - bp_iter: The number of iterations to run belief propagation.

    Returns:
    - fitness: The fitness value for the given solution.
    """
    if last_fitness_result != 0:
        return last_fitness_result
    G = G_from_solution(solution, k)
    H = generate_parity_check_matrix(G)
    ber = test_G(G, sample_size, sigma, snr, H, bp_iter=bp_iter)
    fitness = (1 / (ber + delta)) ** gamma
    return fitness


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
        delta=1e-20,
        gamma=3,
        bp_iter=DEFAULT_BP_MAX_ITER,
    ):
        self.k = k
        self.n = n
        self.num_initial_population = num_initial_population
        self.sample_size = sample_size
        self.num_parents_mating = num_parents_mating
        initial_population_size = (
            self.num_initial_population,
            self.k,
            (self.n - self.k),
        )
        self.population = np.random.choice(a=[0, 1], size=initial_population_size)
        self.channel_func = AWGN_channel
        self.fitness_result = np.zeros(num_initial_population)
        self.best_of_the_bests_sol = np.zeros((self.k, (self.n - self.k)))
        self.offspring_size = offspring_size
        self.p_mutation = p_mutation
        self.num_generations = num_generations
        self.bp_iter = bp_iter

        self.fitness_result_normalize = None
        self.start_generation_time = 0
        self.last_fitness = 0
        self.best_solution = None
        self.best_solution_fitness = 0
        self.sigma = EbN0_to_std(ebn0, self.k / self.n)
        self.snr = EbN0_to_snr(ebn0, self.k / self.n)
        self.bests_nlog = []

        self.delta = delta
        self.gamma = gamma
        self.max_fitness = (1 / self.delta) ** self.gamma
        self.all_nlog = []

    def fitness(self):
        """
        Calculate the fitness of each individual in the population.

        This method applies the `fitness_single` function to each individual in the population
        using multiple processes for parallel execution. The fitness result is stored in the
        `fitness_result` attribute.

        After calculating the fitness, the method normalizes the fitness results, finds the
        index of the best solution, and updates the `best_solution` and `best_solution_fitness`
        attributes accordingly.
        """
        apply_function_with_processes(
            fitness_single,
            self.population,
            self.fitness_result,
            self.k,
            self.sample_size,
            self.sigma,
            self.snr,
            self.delta,
            self.gamma,
            self.bp_iter,
        )

        self.fitness_result_normalize = self.fitness_result / np.sum(
            self.fitness_result
        )
        best_solution_index = np.argmax(self.fitness_result)
        self.best_solution = self.population[best_solution_index].copy()
        self.best_solution_fitness = self.fitness_result[best_solution_index]

    def crossover(self):
        """
        Performs crossover operation on the population.

        Returns:
            offsprings (ndarray): The resulting offsprings after crossover.
        """
        indices_of_parents_mating = np.random.choice(
            self.num_initial_population,
            size=self.num_parents_mating,
            p=self.fitness_result_normalize,
            replace=False,
        )
        indices = np.random.choice(
            self.num_parents_mating, size=(2, self.offspring_size)
        )
        a = self.population[indices_of_parents_mating][indices[0]]
        b = self.population[indices_of_parents_mating][indices[1]]

        random_points = np.random.randint(
            self.k, size=(self.offspring_size, self.n - self.k)
        )
        mask = np.arange(self.k) < random_points[:, :, None]
        mask = mask.transpose((0, 2, 1))
        offsprings = np.where(mask, a, b)

        return offsprings

    def mutation(self, offsprings):
        """
        Applies mutation to the offsprings.

        Args:
            offsprings (ndarray): The offsprings to be mutated.
        """
        mutation_mask = np.random.rand(*offsprings.shape) < self.p_mutation
        offsprings[mutation_mask] ^= 1

    def run(self):
        """
        Runs the genetic algorithm for a specified number of generations.

        This method performs the following steps for each generation:
        1. Calculates the fitness of the current population.
        2. Performs crossover to generate offspring.
        3. Applies mutation to the offspring.
        4. Selects the best offspring to replace the worst individuals in the population.
        5. Ends the current generation and updates necessary variables.
        6. Checks if the best solution has been found and terminates if so.
        """
        self.start_generation_time = time.time()
        for i in range(self.num_generations):
            self.fitness()
            offsprings = self.crossover()
            self.mutation(offsprings)
            argsort_result = np.argsort(self.fitness_result)
            offsprings_indices = argsort_result[: self.offspring_size]
            self.population[offsprings_indices] = offsprings
            self.end_generation(i)
            self.fitness_result[offsprings_indices] = np.zeros(self.offspring_size)
            if self.best_solution_fitness == self.max_fitness:
                break

    def end_generation(self, generations_index):
        """
        Perform end-of-generation operations and print relevant information.
        """
        change_in_fitness = self.best_solution_fitness - self.last_fitness
        G = G_from_solution(self.best_solution, self.k)
        ber = test_G(
            G, self.sample_size * 10, self.sigma, self.snr, bp_iter=self.bp_iter
        )
        nlog = -np.log(ber)
        if self.bests_nlog == [] or nlog < np.min(self.bests_nlog):
            self.best_of_the_bests_sol = self.best_solution
        self.bests_nlog.append(nlog)
        print(f"Generation = {generations_index}")
        print(f"Fitness    = {self.best_solution_fitness}")
        print(f"Change in Fitness     = {change_in_fitness}")
        print(f"BER value of best solution: {ber}")
        print(f"Negative natural logarithm of Bit Error Rate: {nlog}")
        end_generation_time = time.time()
        print(
            f"Generation Running Time: {end_generation_time - self.start_generation_time} [s]"
        )
        self.start_generation_time = end_generation_time
        self.last_fitness = self.best_solution_fitness
        generation_nlog = -np.log(self.fitness_result ** (-1 / self.gamma) - self.delta)
        self.all_nlog.append(generation_nlog)

    @staticmethod
    def load_from_file(filename):
        """
        Load a GeneticAlgorithm object from a file.
        """
        with open(filename, "rb") as file:
            ga = pickle.load(file)
        return ga

    def dump_to_file(self, filename):
        """
        Dump the current object to a file using pickle.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
