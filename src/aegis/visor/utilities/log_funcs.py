import logging


def log_debug(func):
    def wrapper(*args, **kwargs):
        logging.debug(f"Executing function: {func.__name__}.")
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
