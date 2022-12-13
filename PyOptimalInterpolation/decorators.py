
from functools import wraps
import time


def timer(func):
    @wraps(func)
    def caller(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"'{func.__name__}': {t1-t0:.3f} seconds")
        return result
    return caller


if __name__ == '__main__':

    pass
