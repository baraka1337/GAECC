from tqdm import tqdm


from concurrent.futures import ProcessPoolExecutor


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
