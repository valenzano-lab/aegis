import numpy as np

from aegis.modules.reproduction.recombination import recombine
from aegis.modules.reproduction.assortment import assort
from aegis.modules.reproduction.mutation import _mutate_by_bit, _mutate_by_index


class Reproducer:
    """Offspring generator

    Recombines, assorts and mutates genomes of mating individuals to
        create new genomes of their offspring."""

    def __init__(self, RECOMBINATION_RATE, MUTATION_RATIO, REPRODUCTION_MODE, MUTATION_METHOD):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE

        # Mutation rates
        self.rate_0to1 = MUTATION_RATIO / (1 + MUTATION_RATIO)
        self.rate_1to0 = 1 / (1 + MUTATION_RATIO)

        # Set mutation method
        if MUTATION_METHOD == "by_index":
            self._mutate = _mutate_by_index
        elif MUTATION_METHOD == "by_bit":
            self._mutate = _mutate_by_bit
        else:
            raise ValueError("MUTATION_METHOD must be 'by_index' or 'by_bit'")

    def __call__(self, genomes, muta_prob):
        """Exposed method"""

        if self.REPRODUCTION_MODE == "sexual":
            genomes = self.recombine(self.RECOMBINATION_RATE, genomes)
            genomes, _ = self.assort(genomes)

        genomes = self._mutate(self.rate_0to1, self.rate_1to0, genomes, muta_prob)

        return genomes
