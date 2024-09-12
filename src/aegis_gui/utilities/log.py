import logging
import sys


def get_location_of_func(func):
    module_name = func.__module__
    module = sys.modules[module_name]
    module_path = module.__file__
    return f"{module_path}, line {func.__code__.co_firstlineno}"


def log_debug(func):
    def wrapper(*args, **kwargs):
        logging.debug(f"function {func.__name__} ({get_location_of_func(func)})")
        return func(*args, **kwargs)

    return wrapper


def log_info(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Executing function: {func.__name__}.")
        return func(*args, **kwargs)

    return wrapper


def log_error(func):
    def wrapper(*args, **kwargs):
        logging.error(f"Executing function: {func.__name__}.")
        return func(*args, **kwargs)

    return wrapper
