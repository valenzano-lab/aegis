"""Disease manager

Disease status:
    0 .. susceptible
    1 .. infected
    -1 .. dead
"""

import math
from aegis.help import other

BACKGROUND_INFECTIVITY = None
TRANSMISSIBILITY = None
RECOVERY_RATE = None
FATALITY_RATE = None


def init(module, BACKGROUND_INFECTIVITY, TRANSMISSIBILITY, RECOVERY_RATE, FATALITY_RATE):
    module.BACKGROUND_INFECTIVITY = BACKGROUND_INFECTIVITY
    module.TRANSMISSIBILITY = TRANSMISSIBILITY
    module.RECOVERY_RATE = RECOVERY_RATE
    module.FATALITY_RATE = FATALITY_RATE


def get_infection_probability(infection_density):
    return BACKGROUND_INFECTIVITY - 0.5 + 1 / (1 + math.exp(-TRANSMISSIBILITY * infection_density))


def kill(population):
    """
    First try infecting susceptible.
    """
    probs = other.rng.random(len(population), dtype=float)

    # current status
    infected = population.disease == 1
    susceptible = population.disease == 0

    # compute infection probability
    infection_density = infected.sum() / len(population)
    infection_probability = get_infection_probability(infection_density=infection_density)

    # recoveries from old infections
    population.disease[infected & (probs < RECOVERY_RATE)] = 0

    # fatalities
    # overrides recoveries
    population.disease[infected & (probs < FATALITY_RATE)] = -1

    # new infections
    population.disease[susceptible & (probs < infection_probability)] = 1
