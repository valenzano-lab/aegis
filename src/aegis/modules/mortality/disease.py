"""Disease manager

Disease status:
    0 .. susceptible
    1 .. infected
    -1 .. dead
"""

import math
from aegis import cnf
from aegis import var


def get_infection_probability(infection_density):
    return cnf.BACKGROUND_INFECTIVITY - 0.5 + 1 / (1 + math.exp(-cnf.TRANSMISSIBILITY * infection_density))


def kill(population):
    """
    First try infecting susceptible.
    """

    if len(population) == 0:
        return

    probs = var.rng.random(len(population), dtype=float)

    # current status
    infected = population.disease == 1
    susceptible = population.disease == 0

    # compute infection probability
    infection_density = infected.sum() / len(population)
    infection_probability = get_infection_probability(infection_density=infection_density)

    # recoveries from old infections
    population.disease[infected & (probs < cnf.RECOVERY_RATE)] = 0

    # fatalities
    # overrides recoveries
    population.disease[infected & (probs < cnf.FATALITY_RATE)] = -1

    # new infections
    population.disease[susceptible & (probs < infection_probability)] = 1
