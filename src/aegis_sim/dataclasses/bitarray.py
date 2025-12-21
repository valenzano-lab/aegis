"""Abstract away genomes."""

import numpy as np


class BitArray:
    """Base class for pseudogenomic-like data structures."""

    def __init__(self, array, dtype):
        self.array = array.astype(dtype)

    def __len__(self):
        return len(self.array)

    def flatten(self):
        return self.array.reshape(len(self), -1)

    def get(self, individuals):
        return self.array[individuals]

    def add(self, other):
        self.array = np.concatenate([self.array, other.array])

    def keep(self, individuals):
        self.array = self.array[individuals]

    def get_array(self):
        return self.array.copy()

    def shape(self):
        return self.array.shape


class Genomes(BitArray):
    # TODO add logicalxor
    dtype = np.bool_

    def __init__(self, array):
        super().__init__(array, self.dtype)

    def __getitem__(self, key):
        new_array = self.array[key]
        return Genomes(new_array)


class Origins(BitArray):
    dtype = np.uint8
    
    def __init__(self, array):
        super().__init__(array, self.dtype)

    def __getitem__(self, key):
        new_array = self.array[key]
        return Origins(new_array)
