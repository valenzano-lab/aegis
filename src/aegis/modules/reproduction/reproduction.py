"""Offspring generator

Recombines, assorts and mutates genomes of mating individuals to
    create new genomes of their offspring."""
from aegis.modules.reproduction import recombination, assortment, mutation

RECOMBINATION_RATE = None
MUTATION_RATE = None
REPRODUCTION_MODE = None
MUTATION_METHOD = None
_mutate = None
rate_0to1 = None
rate_1to0 = None


def init(self, RECOMBINATION_RATE, MUTATION_RATIO, REPRODUCTION_MODE, MUTATION_METHOD):
    recombination.RECOMBINATION_RATE = RECOMBINATION_RATE

    # Mutation rates
    self.rate_0to1 = MUTATION_RATIO / (1 + MUTATION_RATIO)
    self.rate_1to0 = 1 / (1 + MUTATION_RATIO)


    # Set mutation method
    if MUTATION_METHOD == "by_index":
        self._mutate = mutation._mutate_by_index
    elif MUTATION_METHOD == "by_bit":
        self._mutate = mutation._mutate_by_bit
    else:
        raise ValueError("MUTATION_METHOD must be 'by_index' or 'by_bit'")


def do(genomes, muta_prob):
    """Exposed method"""

    if REPRODUCTION_MODE == "sexual":
        genomes = recombination.do(genomes)
        genomes, _ = assortment.do(genomes)

    genomes = _mutate(rate_0to1, rate_1to0, genomes, muta_prob)

    return genomes
