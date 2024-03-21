"""Modifier of fitness landscape

Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
"""

import numpy as np
from aegis.hermes import hermes


class Flipmap:
    def __init__(self, FLIPMAP_CHANGE_RATE, genome_shape):
        self.FLIPMAP_CHANGE_RATE = FLIPMAP_CHANGE_RATE

        if self.FLIPMAP_CHANGE_RATE == 0:
            self.map = None
        else:
            self.map = np.zeros(genome_shape, dtype=np.bool_)

    def evolve(self, stage):
        """Modify the flipmap"""
        if (self.map is None) or (stage % self.FLIPMAP_CHANGE_RATE > 0):
            return

        indices = tuple(hermes.rng.integers(self.map.shape))
        self.map[indices] = ~self.map[indices]

    def call(self, array):
        """Return the genomes reinterpreted"""
        if self.map is not None:
            array = np.logical_xor(self.map, array)
        return array
