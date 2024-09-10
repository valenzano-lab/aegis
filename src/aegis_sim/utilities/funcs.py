import time
import logging

from aegis_sim import variables
from aegis_sim.parameterization import parametermanager


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


def skip(rate_name) -> bool:
    """Should you skip an action performed at a certain rate"""

    rate = getattr(parametermanager.parameters, rate_name)

    # Skip if rate deactivated
    if rate <= 0:
        return True

    # Do not skip first step
    if variables.steps == 1:
        return False

    # Skip unless step is divisible by rate
    return variables.steps % rate > 0


def steps_to_end() -> int:
    return parametermanager.parameters.STEPS_PER_SIMULATION - variables.steps
