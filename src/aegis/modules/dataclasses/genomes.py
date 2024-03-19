"""Abstract away genomes."""

import numpy as np


class Genomes:
    def __init__(self, array):
        self.array = array.astype(np.bool_)

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

    def get_array(self):
        return self.array.copy()

    # TODO add logicalxor
