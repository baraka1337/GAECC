from tqdm import tqdm


from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt


def apply_function_with_processes(func, iterator, result, *args, **kwargs):
    """
    Apply a function with processes to each item in an iterator using ProcessPoolExecutor.

    Args:
    - iterator: Iterator containing items to apply the function on.
    - func: Function to be applied on each item.
    - *args: Extra positional arguments to be passed to the function.
    - **kwargs: Extra keyword arguments to be passed to the function.
    """
    with ProcessPoolExecutor() as executor:
        # Map each item in the iterator to the function with given arguments
        futures = {
            executor.submit(func, item, result[i], *args, **kwargs): (i, item)
            for i, item in enumerate(iterator)
        }

        # Wait for all threads to finish
        for i, future in tqdm(enumerate(futures)):
            value = future.result()
            result[i] = value


def plot_all_generations(ga, name):
    plt.figure()
    for i, nlog in enumerate(ga.all_nlog):
        if i == 0 or (i + 1) % 5 == 0:
            plt.plot(sorted(nlog), label=f"gen {i+1}")
    plt.grid()
    plt.xlabel("Population")
    plt.ylabel("-Log(ber)")
    plt.legend()
    plt.savefig(f"{name}_all.png")


def plot_best(ga, name):
    best_of_the_bests = np.argmax(ga.bests_nlog)
    plt.figure()
    plt.plot(ga.bests_nlog)
    plt.plot(
        best_of_the_bests,
        ga.bests_nlog[best_of_the_bests],
        "o",
        color="red",
        label=f"Best: {ga.bests_nlog[best_of_the_bests]:.2f}",
    )
    plt.xlabel("Generation")
    plt.ylabel("-Log(ber)")
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(f"{name}_best.png")
