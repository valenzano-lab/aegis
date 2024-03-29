"""Overshoot resolver

Decides which individuals to eliminate when there is overcrowding.
"""

import numpy as np
from aegis.hermes import hermes


class Starvation:
    def __init__(
        self,
        STARVATION_RESPONSE,
        STARVATION_MAGNITUDE,
        CLIFF_SURVIVORSHIP,
        MAX_POPULATION_SIZE,
    ):

        self.STARVATION_MAGNITUDE = STARVATION_MAGNITUDE
        self.CLIFF_SURVIVORSHIP = CLIFF_SURVIVORSHIP
        self.MAX_POPULATION_SIZE = MAX_POPULATION_SIZE

        self.consecutive_overshoot_n = 0  # For starvation mode

        self.func = {
            "treadmill_random": self._treadmill_random,
            "treadmill_boomer": self._treadmill_boomer,
            "treadmill_zoomer": self._treadmill_zoomer,
            "cliff": self._cliff,
            "gradual": self._gradual,
            "logistic": self._logistic,
        }[STARVATION_RESPONSE]

    def __call__(self, n):
        """Exposed method"""
        global consecutive_overshoot_n  # TODO do not use global
        if n <= self.MAX_POPULATION_SIZE:
            consecutive_overshoot_n = 0
            return np.zeros(n, dtype=np.bool_)
        else:
            consecutive_overshoot_n += 1
            return self.func(n)

    def _logistic(self, n):
        """Kill random individuals with logistic-like probability."""
        ratio = n / self.MAX_POPULATION_SIZE

        # when ratio == 1, kill_probability is set to 0
        # when ratio == 2, kill_probability is set to >0
        kill_probability = 2 / (1 + np.exp(-ratio + 1)) - 1

        random_probabilities = hermes.rng.random(n, dtype=np.float32)
        mask = random_probabilities < kill_probability
        return mask

    def _gradual(self, n):
        """Kill random individuals with time-increasing probability.

        The choice of individuals is random.
        The probability of dying increases each consecutive stage of overcrowding.
        The probability of dying resets to the base value once the population dips under the maximum allowed size.
        """
        surv_probability = (1 - self.STARVATION_MAGNITUDE) ** consecutive_overshoot_n
        random_probabilities = hermes.rng.random(n, dtype=np.float32)
        mask = random_probabilities > surv_probability
        return mask

    def _treadmill_random(self, n):
        """Kill random individuals.

        The population size is brought down to the maximum allowed size in one go.
        """
        indices = hermes.rng.choice(n, n - self.MAX_POPULATION_SIZE, replace=False)
        mask = np.zeros(n, dtype=np.bool_)
        mask[indices] = True
        return mask

    def _cliff(self, n):
        """Kill all individuals except a small random proportion.

        The proportion is defined as the parameter CLIFF_SURVIVORSHIP.
        This function will not necessarily bring the population below the maximum allowed size.
        """
        indices = hermes.rng.choice(
            n,
            int(self.MAX_POPULATION_SIZE * self.CLIFF_SURVIVORSHIP),
            replace=False,
        )
        mask = np.ones(n, dtype=np.bool_)
        mask[indices] = False
        return mask

    def _treadmill_boomer(self, n):
        """Kill the oldest individuals.

        The population size is brought down to the maximum allowed size in one go.

        NOTE: Why `-self.MAX_POPULATION_SIZE :`? Because old individuals are at the beginning of the population array.
        """
        mask = np.ones(n, dtype=np.bool_)
        mask[-self.MAX_POPULATION_SIZE :] = False
        return mask

    def _treadmill_zoomer(self, n):
        """Kill the youngest individuals.

        The population size is brought down to the maximum allowed size in one go.

        NOTE: Why `: self.MAX_POPULATION_SIZE`? Because young individuals are appended to the end of the population array.
        """
        mask = np.ones(n, dtype=np.bool_)
        mask[: self.MAX_POPULATION_SIZE] = False
        return mask
