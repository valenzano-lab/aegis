"""Overshoot resolver

Decides which individuals to eliminate when there is overcrowding.
"""

import numpy as np
from aegis.pan import var
from aegis.pan import cnf

func = None  # defined below
consecutive_overshoot_n = 0  # For starvation mode


def call(n):
    """Exposed method"""
    global consecutive_overshoot_n  # TODO do not use global
    if n <= cnf.MAX_POPULATION_SIZE:
        consecutive_overshoot_n = 0
        return np.zeros(n, dtype=np.bool_)
    else:
        consecutive_overshoot_n += 1
        return func(n)


def _logistic(n):
    """Kill random individuals with logistic-like probability."""
    ratio = n / cnf.MAX_POPULATION_SIZE

    # when ratio == 1, kill_probability is set to 0
    # when ratio == 2, kill_probability is set to >0
    kill_probability = 2 / (1 + np.exp(-ratio + 1)) - 1

    random_probabilities = var.rng.random(n, dtype=np.float32)
    mask = random_probabilities < kill_probability
    return mask


def _gradual(n):
    """Kill random individuals with time-increasing probability.

    The choice of individuals is random.
    The probability of dying increases each consecutive stage of overcrowding.
    The probability of dying resets to the base value once the population dips under the maximum allowed size.
    """
    surv_probability = (1 - cnf.STARVATION_MAGNITUDE) ** consecutive_overshoot_n
    random_probabilities = var.rng.random(n, dtype=np.float32)
    mask = random_probabilities > surv_probability
    return mask


def _treadmill_random(n):
    """Kill random individuals.

    The population size is brought down to the maximum allowed size in one go.
    """
    indices = var.rng.choice(n, n - cnf.MAX_POPULATION_SIZE, replace=False)
    mask = np.zeros(n, dtype=np.bool_)
    mask[indices] = True
    return mask


def _cliff(n):
    """Kill all individuals except a small random proportion.

    The proportion is defined as the parameter CLIFF_SURVIVORSHIP.
    This function will not necessarily bring the population below the maximum allowed size.
    """
    indices = var.rng.choice(
        n,
        int(cnf.MAX_POPULATION_SIZE * cnf.CLIFF_SURVIVORSHIP),
        replace=False,
    )
    mask = np.ones(n, dtype=np.bool_)
    mask[indices] = False
    return mask


def _treadmill_boomer(n):
    """Kill the oldest individuals.

    The population size is brought down to the maximum allowed size in one go.
    
    NOTE: Why `-cnf.MAX_POPULATION_SIZE :`? Because old individuals are at the beginning of the population array.
    """
    mask = np.ones(n, dtype=np.bool_)
    mask[-cnf.MAX_POPULATION_SIZE :] = False
    return mask


def _treadmill_zoomer(n):
    """Kill the youngest individuals.

    The population size is brought down to the maximum allowed size in one go.
    
    NOTE: Why `: cnf.MAX_POPULATION_SIZE`? Because young individuals are appended to the end of the population array.
    """
    mask = np.ones(n, dtype=np.bool_)
    mask[: cnf.MAX_POPULATION_SIZE] = False
    return mask


func = {
    "treadmill_random": _treadmill_random,
    "treadmill_boomer": _treadmill_boomer,
    "treadmill_zoomer": _treadmill_zoomer,
    "cliff": _cliff,
    "gradual": _gradual,
    "logistic": _logistic,
}[cnf.STARVATION_RESPONSE]