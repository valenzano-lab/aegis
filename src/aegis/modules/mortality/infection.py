"""Infection manager

Infection status:
    0 .. susceptible
    1 .. infected
    -1 .. dead
"""

import math
from aegis.hermes import hermes


class Infection:
    """

    GUI
    Infection is an optional source of mortality.
    [[FATALITY_RATE]] specifies how deadly the infection is; thus if set to 0, no deaths from infection will occur.
    The infection modeling is inspired by the SIR (susceptible-infectious-removed) model.

    Individuals cannot gain immunity, thus can get reinfected many times.
    The probability to die from an infection is constant as long as the individual is infected; there is no incubation period nor disease progression.
    The same is true for recovering from the disease, which is equal to [[RECOVERY_RATE]].

    Both of these are independent of age and genetics.

    The infectious agent can be transmitted from an individual to an individual but can also be contracted from the environment (and can therefore not be fully eradicated).
    The probability to acquire the infection from the environment is equal to [[BACKGROUND_INFECTIVITY]], and from other infected individuals it grows with [[TRANSMISSIBILITY]]
    but also (logistically) with the proportion of the infected population.
    """

    def __init__(self, BACKGROUND_INFECTIVITY, TRANSMISSIBILITY, RECOVERY_RATE, FATALITY_RATE):
        self.BACKGROUND_INFECTIVITY = BACKGROUND_INFECTIVITY
        self.TRANSMISSIBILITY = TRANSMISSIBILITY
        self.RECOVERY_RATE = RECOVERY_RATE
        self.FATALITY_RATE = FATALITY_RATE

    def get_infection_probability(self, infection_density):
        return self.BACKGROUND_INFECTIVITY - 0.5 + 1 / (1 + math.exp(-self.TRANSMISSIBILITY * infection_density))

    def __call__(self, population):
        """
        First try infecting susceptible.
        """

        # TODO Do not change population here, only return new status

        if len(population) == 0:
            return

        probs = hermes.rng.random(len(population), dtype=float)

        # current status
        infected = population.infection == 1
        susceptible = population.infection == 0

        # compute infection probability
        infection_density = infected.sum() / len(population)
        infection_probability = self.get_infection_probability(infection_density=infection_density)

        # recoveries from old infections
        population.infection[infected & (probs < self.RECOVERY_RATE)] = 0

        # fatalities
        # overrides recoveries
        population.infection[infected & (probs < self.FATALITY_RATE)] = -1

        # new infections
        population.infection[susceptible & (probs < infection_probability)] = 1
