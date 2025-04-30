"""Overshoot resolver

Decides which individuals to eliminate when there is overcrowding.
"""

import logging
import numpy as np
from aegis_sim.submodels.frailty import frailty
from aegis_sim import variables


class Starvation:
    """

    GUI
    Starvation is a source of mortality useful for modeling death from lack of resources.
    It is usually the largest contributor to mortality. Generally, each individual requires one unit of resources to survive.
    When the population size exceeds the amount of resources available, the population experiences starvation.
    Starvation is either experienced by the whole population or by no individual; i.e. the distribution of resources is equal across individuals.

    Starvation mortality can operate under one of two modes. Under the first mode, population is sensitive to the
    amount of resource deficit it is experiencing and will respond promptly to it. For example, if the population size is 1000
    and there are only 800 resource units available, approximately only 800 individuals will survive, thus approximately 200 will die,
    so the mortality will be about 20% (200/1000).

    Under the second mode, population is not sensitive to the amount of resource deficit but rather to the number of consecutive simulation steps
    that it has experienced starvation. When it first experiences resource deficit, the mortality will be [[STARVATION_MORTALITY_FACTOR]].
    If in the next step, the population size is still greater than the amount of available resources, the mortality will now be
    approximately twice as high; in the next step, about three times, etc.

    The probability to die is independent of genetics (genetics do not confer protection nor susceptibility to starvation).
    However, age can modify the probability to die, depending on the [[FRAILTY_MODIFIER]]. When [[FRAILTY_MODIFIER]] is 0,
    there is no age-dependent effect. When it is greater than 0, then starvation mortality is magnified by [[FRAILTY_MODIFIER]]
    for the oldest age class, by 0 for the youngest age class, and proportionally for the intermediate age classes (e.g. by 20% of the
    [[FRAILTY_MODIFIER]] for the age class 10 if [[AGE_LIMIT]] is 50 because 10/50=0.2).

    If [[STARVATION_MORTALITY_MAXIMUM]] is set, the final computed mortality will be at maximum of that value, not higher.

    Note that if the species is oviparious ([[INCUBATION_PERIOD]]), the produced eggs do not consume resources and are
    immune to starvation mortality (until they hatch).
    """

    def init(self, STARVATION_MORTALITY_MAXIMUM, STARVATION_MORTALITY_FACTOR):
        self.STARVATION_MORTALITY_MAXIMUM = STARVATION_MORTALITY_MAXIMUM
        self.STARVATION_MORTALITY_FACTOR = STARVATION_MORTALITY_FACTOR
        self.consecutive_overshoot_n = 0

    def get_mask_kill(self, ages, resources_scavenged):
        population_size = len(ages)

        # No starvation
        if population_size <= resources_scavenged:
            self.consecutive_overshoot_n = 0
            return np.zeros(population_size, dtype=np.bool_)
        else:
            self.consecutive_overshoot_n += 1

        # Compute mortality
        if self.STARVATION_MORTALITY_FACTOR is None:
            # If mortality depends on the resource deficit
            # This mode will prevent over- and undercorrections
            mortality = 1 - resources_scavenged / population_size
            assert 0 < mortality <= 1
        else:
            # If mortality depends on the number of consecutive steps experiencing starvation
            survival = (1 - self.STARVATION_MORTALITY_FACTOR) ** self.consecutive_overshoot_n
            mortality = 1 - survival

        # Consider age
        mortalities = frailty.modify(hazard=mortality, ages=ages)

        # Restrict the upper bound of mortality
        mortalities[mortalities > self.STARVATION_MORTALITY_MAXIMUM] = self.STARVATION_MORTALITY_MAXIMUM

        # Compute mortality mask
        random_probabilities = variables.rng.random(population_size)
        mask = random_probabilities < mortalities

        return mask

    # @staticmethod
    # def _logistic(n, resource_availability):
    #     """Kill random individuals with logistic-like probability."""
    #     ratio = n / resource_availability

    #     # when ratio == 1, kill_probability is set to 0
    #     # when ratio == 2, kill_probability is set to >0
    #     kill_probability = 2 / (1 + np.exp(-ratio + 1)) - 1

    #     random_probabilities = variables.rng.random(n)
    #     mask = random_probabilities < kill_probability
    #     return mask

    # def _gradual(self, n, resource_availability):
    #     """Kill random individuals with time-increasing probability.

    #     The choice of individuals is random.
    #     The probability of dying increases each consecutive step of overcrowding.
    #     The probability of dying resets to the base value once the population dips under the maximum allowed size.
    #     """
    #     surv_probability = (1 - self.STARVATION_MAGNITUDE) ** self.consecutive_overshoot_n
    #     random_probabilities = variables.rng.random(n)
    #     mask = random_probabilities > surv_probability
    #     return mask

    # def _worsening_proportional(self, n, resource_availability):
    #     """Kill random individuals with time-increasing probability.

    #     The choice of individuals is random.
    #     The probability of dying increases each consecutive step of overcrowding.
    #     The probability of dying resets to the base value once the population dips under the maximum allowed size.
    #     """
    #     surv_probability = (resource_availability / n) ** self.consecutive_overshoot_n
    #     f = frailty.modify(1 - surv_probability, 10)
    #     random_probabilities = variables.rng.random(n)
    #     mask = random_probabilities > surv_probability
    #     return mask

    # @staticmethod
    # def _treadmill_random(n, resource_availability):
    #     """Kill random individuals.

    #     The population size is brought down to the maximum allowed size in one go.
    #     """
    #     indices = variables.rng.choice(n, n - int(resource_availability), replace=False)
    #     mask = np.zeros(n, dtype=np.bool_)
    #     mask[indices] = True
    #     return mask

    # # def _cliff(self, n, resource_availability):
    # #     """Kill all individuals except a small random proportion.

    # #     The proportion is defined as the parameter CLIFF_SURVIVORSHIP.
    # #     This function will not necessarily bring the population below the maximum allowed size.
    # #     """
    # #     indices = variables.rng.choice(
    # #         n,
    # #         int(resource_availability * self.CLIFF_SURVIVORSHIP),
    # #         replace=False,
    # #     )
    # #     mask = np.ones(n, dtype=np.bool_)
    # #     mask[indices] = False
    # #     return mask

    # @staticmethod
    # def _treadmill_boomer(n, resource_availability):
    #     """Kill the oldest individuals.

    #     The population size is brought down to the maximum allowed size in one go.

    #     NOTE: Why `-resource_availability :`? Because old individuals are at the beginning of the population array.
    #     """
    #     mask = np.ones(n, dtype=np.bool_)
    #     mask[-int(resource_availability) :] = False
    #     return mask

    # @staticmethod
    # def _treadmill_zoomer(n, resource_availability):
    #     """Kill the youngest individuals.

    #     The population size is brought down to the maximum allowed size in one go.

    #     NOTE: Why `: resource_availability`? Because young individuals are appended to the end of the population array.
    #     """
    #     mask = np.ones(n, dtype=np.bool_)
    #     mask[: int(resource_availability)] = False
    #     return mask

    # def _treadmill_boomer_soft(self, n, resource_availability):
    #     """Kill older individuals more.
    #     Old individuals are positioned more to the front of the array.
    #     True in the mask means death.
    #     """
    #     mask = self._treadmill_soft(1, 0, n, resource_availability)
    #     return mask

    # def _treadmill_zoomer_soft(self, n, resource_availability):
    #     """Kill younger individuals more.
    #     Young individuals are positioned later in the array.
    #     True in the mask means death."""
    #     mask = self._treadmill_soft(0, 1, n, resource_availability)
    #     return mask

    # @staticmethod
    # def _treadmill_soft(linspace_from, linspace_to, n, resource_availability):
    #     """
    #     Young individuals are positioned later in the array; older earlier.
    #     True in the mask means death.
    #     """
    #     p = np.linspace(linspace_from, linspace_to, n) ** 5  # **5 to make it superlinear
    #     p /= p.sum()  # ensure sum(p) is 1

    #     mask = np.zeros(n, dtype=np.bool_)

    #     a = np.arange(n)
    #     indices_dead = variables.rng.choice(a, size=n - int(resource_availability), p=p, replace=False)
    #     mask[indices_dead] = True
    #     return mask


starvation = Starvation()
