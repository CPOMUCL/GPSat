
from functools import wraps
import time


def timer(func):
    """
    This function is a decorator that can be used to time the execution of other functions.

    It takes a function as an argument and returns a new function that wraps the original function.

    When the wrapped function is called, it measures the time it takes to execute the original function and
    prints the result to the console.

    The function uses the `time.perf_counter()` function to measure the time.
    This function returns the current value of a performance counter, which is a high-resolution timer
    that measures the time in seconds since a fixed point in time.

    The wrapped function takes any number of positional and keyword arguments,
    which are passed on to the original function.
    The result of the original function is returned by the wrapped function.

    The decorator also uses the `functools.wraps()` function to preserve the metadata of the
    original function, such as its name, docstring, and annotations.
    This makes it easier to debug and introspect the code.

    To use the decorator, simply apply it to the function you want to time, like this:

    @timer
    def my_function():


    """
    @wraps(func)
    def caller(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        # TODO: could check kwargs for verbose and decide not print
        print(f"'{func.__name__}': {t1-t0:.3f} seconds")
        return result
    return caller


if __name__ == '__main__':

    pass
