"""Abstract away genomes."""

import numpy as np

from aegis.pan import cnf

from aegis.modules.genetics.reproduction import recombination, assortment, mutation


class Genomes:
    pass

    def __init__(self, array):
        self.array = array.astype(np.bool_)

    # def __getitem__(self, mask):
    #     return self.array[mask]

    def __len__(self):
        return len(self.array)

    def flatten(self):
        return self.array.reshape(len(self), -1)

    def get(self, individuals):
        return self.array[individuals]

    def add(self, genomes):
        self.array = np.concatenate([self.array, genomes.array])

    def keep(self, individuals):
        self.array = self.array[individuals]

    def generate_offspring_genomes(self, genomes, muta_prob):

        if cnf.REPRODUCTION_MODE == "sexual":
            genomes = recombination.do(genomes)
            genomes, _ = assortment.do(Genomes(genomes))

        genomes = mutation.mutator._mutate(genomes, muta_prob)
        genomes = Genomes(genomes)
        return genomes

    def get_array(self):
        return self.array.copy()


# TODO add logicalxor
