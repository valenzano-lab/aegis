"""Overshoot resolver

Decides which individuals to eliminate when there is overcrowding.
"""

import numpy as np


class Starvation:
    """

    GUI
    Starvation is an obligatory source of mortality, useful for modeling death from lack of resources.
    The parameter [[CARRYING_CAPACITY]] specifies the amount of resources.
    Generally, each individual requires one unit of resources; otherwise, they are at risk of starvation.
    When population size exceeds [[CARRYING_CAPACITY]], random individuals will start dying.

    The probability to die is genetics-independent (genetics do not confer protection or susceptibility to starvation).
    However, age can modify the probability to die, depending on the [[STARVATION_RESPONSE]].
    When [[STARVATION_RESPONSE]] is set to treadmill_zoomer, young individuals will start dying first;
    for treadmill_boomer, older individuals die first.

    Under other [[STARVATION_RESPONSE]]s, starvation affects all ages equally, but the dynamics of starvation are different.
    When response is set to gradual, death from starvation is at first low, but increases with each subsequent
    step of insufficient resources (the speed of increase is parameterized by [[STARVATION_MAGNITUDE]]).
    When response is set to treadmill_random, whenever population exceeds the [[CARRYING_CAPACITY]], it is immediately
    and precisely cut down to [[CARRYING_CAPACITY]]. In contrast, when response is set to cliff,
    whenever [[CARRYING_CAPACITY]] is exceeded, the population is cut down to a fraction of the [[CARRYING_CAPACITY]];
    the fraction is specified by the [[CLIFF_SURVIVORSHIP]] parameter.

    Note that if the species is oviparious ([[INCUBATION_PERIOD]]), the produced eggs do not consume resources and are
    immune to starvation mortality (until they hatch).
    """

    def __init__(
        self,
        STARVATION_RESPONSE,
        STARVATION_MAGNITUDE,
        CLIFF_SURVIVORSHIP,
        CARRYING_CAPACITY,
    ):

        self.STARVATION_MAGNITUDE = STARVATION_MAGNITUDE
        self.CLIFF_SURVIVORSHIP = CLIFF_SURVIVORSHIP
        self.CARRYING_CAPACITY = CARRYING_CAPACITY

        self.consecutive_overshoot_n = 0  # For starvation mode

        self.func = {
            # "sensitive": self._sensitive,
            "treadmill_random": self._treadmill_random,
            "treadmill_boomer": self._treadmill_boomer,
            "treadmill_zoomer": self._treadmill_zoomer,
            "treadmill_boomer_soft": self._treadmill_boomer_soft,
            "treadmill_zoomer_soft": self._treadmill_zoomer_soft,
            "cliff": self._cliff,
            "gradual": self._gradual,
            "logistic": self._logistic,
        }[STARVATION_RESPONSE]

    def __call__(self, n, resource_availability):
        """Exposed method

        False in `mask` means survival. True in `mask` means death.
        """
        # TODO fix resource availability and carrying capacity logic
        if self.func is not self._gradual:
            resource_availability = self.CARRYING_CAPACITY

        if n <= resource_availability:
            self.consecutive_overshoot_n = 0
            return np.zeros(n, dtype=np.bool_)
        else:
            self.consecutive_overshoot_n += 1
            return self.func(n, resource_availability)

    # def _sensitive(self, n, resource_availability):
    #     ratio = n /resource_availability

    def _logistic(self, n, resource_availability):
        """Kill random individuals with logistic-like probability."""
        ratio = n / resource_availability

        # when ratio == 1, kill_probability is set to 0
        # when ratio == 2, kill_probability is set to >0
        kill_probability = 2 / (1 + np.exp(-ratio + 1)) - 1

        random_probabilities = np.random.random(n)
        mask = random_probabilities < kill_probability
        return mask

    def _gradual(self, n, resource_availability):
        """Kill random individuals with time-increasing probability.

        The choice of individuals is random.
        The probability of dying increases each consecutive step of overcrowding.
        The probability of dying resets to the base value once the population dips under the maximum allowed size.
        """
        surv_probability = (1 - self.STARVATION_MAGNITUDE) ** self.consecutive_overshoot_n
        random_probabilities = np.random.random(n)
        mask = random_probabilities > surv_probability
        return mask

    def _treadmill_random(self, n, resource_availability):
        """Kill random individuals.

        The population size is brought down to the maximum allowed size in one go.
        """
        indices = np.random.choice(n, n - int(resource_availability), replace=False)
        mask = np.zeros(n, dtype=np.bool_)
        mask[indices] = True
        return mask

    def _cliff(self, n, resource_availability):
        """Kill all individuals except a small random proportion.

        The proportion is defined as the parameter CLIFF_SURVIVORSHIP.
        This function will not necessarily bring the population below the maximum allowed size.
        """
        indices = np.random.choice(
            n,
            int(resource_availability * self.CLIFF_SURVIVORSHIP),
            replace=False,
        )
        mask = np.ones(n, dtype=np.bool_)
        mask[indices] = False
        return mask

    def _treadmill_boomer(self, n, resource_availability):
        """Kill the oldest individuals.

        The population size is brought down to the maximum allowed size in one go.

        NOTE: Why `-resource_availability :`? Because old individuals are at the beginning of the population array.
        """
        mask = np.ones(n, dtype=np.bool_)
        mask[-int(resource_availability) :] = False
        return mask

    def _treadmill_zoomer(self, n, resource_availability):
        """Kill the youngest individuals.

        The population size is brought down to the maximum allowed size in one go.

        NOTE: Why `: resource_availability`? Because young individuals are appended to the end of the population array.
        """
        mask = np.ones(n, dtype=np.bool_)
        mask[: int(resource_availability)] = False
        return mask

    def _treadmill_boomer_soft(self, n, resource_availability):
        """Kill older individuals more.
        Old individuals are positioned more to the front of the array.
        True in the mask means death.
        """
        mask = self._treadmill_soft(1, 0, n, self.CARRYING_CAPACITY)
        return mask

    def _treadmill_zoomer_soft(self, n, resource_availability):
        """Kill younger individuals more.
        Young individuals are positioned later in the array.
        True in the mask means death."""
        mask = self._treadmill_soft(0, 1, n, self.CARRYING_CAPACITY)
        return mask

    @staticmethod
    def _treadmill_soft(linspace_from, linspace_to, n, CARRYING_CAPACITY):
        """
        Young individuals are positioned later in the array; older earlier.
        True in the mask means death.
        """
        p = np.linspace(linspace_from, linspace_to, n) ** 5  # **5 to make it superlinear
        p /= p.sum()  # ensure sum(p) is 1

        mask = np.zeros(n, dtype=np.bool_)

        a = np.arange(n)
        indices_dead = np.random.choice(a, size=n - int(CARRYING_CAPACITY), p=p, replace=False)
        mask[indices_dead] = True
        return mask
