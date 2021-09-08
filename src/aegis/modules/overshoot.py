
import numpy as np

from aegis.panconfiguration import pan


class Overshoot:
    """Overshoot resolver
    
    Decides which individuals to eliminate when there is overcrowding.
    """

    def __init__(self, OVERSHOOT_EVENT, MAX_POPULATION_SIZE, CLIFF_SURVIVORSHIP):
        self.MAX_POPULATION_SIZE = MAX_POPULATION_SIZE
        self.CLIFF_SURVIVORSHIP = CLIFF_SURVIVORSHIP
        self.func = {
            "treadmill_random": self._treadmill_random,
            "treadmill_boomer": self._treadmill_boomer,
            "treadmill_zoomer": self._treadmill_zoomer,
            "cliff": self._cliff,
            "starvation": self._starvation,
        }[OVERSHOOT_EVENT]

        self.consecutive_overshoot_n = 0  # For starvation mode

    def __call__(self, n):
        """Exposed method"""
        if n <= self.MAX_POPULATION_SIZE:
            self.consecutive_overshoot_n = 0
            return np.zeros(n, bool)
        else:
            self.consecutive_overshoot_n += 1
            return self.func(n)

    def _starvation(self, n):
        """Kill random individuals with time-increasing probability.

        The choice of individuals is random.
        The probability of dying increases each consecutive stage of overcrowding.
        The probability of dying resets to the base value once the population dips under the maximum allowed size.
        """
        surv_probability = 0.95 ** self.consecutive_overshoot_n
        random_probabilities = pan.rng.random(n)
        mask = random_probabilities > surv_probability
        return mask

    def _treadmill_random(self, n):
        """Kill random individuals.

        The population size is brought down to the maximum allowed size in one go.
        """
        indices = pan.rng.choice(n, n - self.MAX_POPULATION_SIZE, replace=False)
        mask = np.zeros(n, bool)
        mask[indices] = True
        return mask

    def _cliff(self, n):
        """Kill all individuals except a small random proportion.

        The proportion is defined as the parameter CLIFF_SURVIVORSHIP.
        This function will not necessarily bring the population below the maximum allowed size.
        """
        indices = pan.rng.choice(
            n,
            int(self.MAX_POPULATION_SIZE * self.CLIFF_SURVIVORSHIP),
            replace=False,
        )
        mask = np.ones(n, bool)
        mask[indices] = False
        return mask

    def _treadmill_boomer(self, n):
        """Kill the oldest individuals.

        The population size is brought down to the maximum allowed size in one go.
        """
        mask = np.ones(n, bool)
        mask[-self.MAX_POPULATION_SIZE :] = False
        return mask

    def _treadmill_zoomer(self, n):
        """Kill the youngest individuals.

        The population size is brought down to the maximum allowed size in one go.
        """
        mask = np.ones(n, bool)
        mask[: self.MAX_POPULATION_SIZE] = False
        return mask
