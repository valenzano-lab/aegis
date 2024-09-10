import time
import logging


# Decorators
def profile_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        logging.info(f"{func.__name__} took {execution_time:.6f} ms to execute")
        return result

    return wrapper
