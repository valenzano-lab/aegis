"""Modifier of fitness landscape

Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
"""

import numpy as np
from aegis.pan import var


class Flipmap:
    def init(self, FLIPMAP_CHANGE_RATE, gstruc_shape):
        self.FLIPMAP_CHANGE_RATE = FLIPMAP_CHANGE_RATE

        if self.FLIPMAP_CHANGE_RATE == 0:
            self.map = None
        else:
            self.map = np.zeros(gstruc_shape, dtype=np.bool_)

    def evolve(self):
        """Modify the flipmap"""
        if (self.map is None) or (var.stage % self.FLIPMAP_CHANGE_RATE > 0):
            return

        indices = tuple(var.rng.integers(self.map.shape))
        self.map[indices] = ~self.map[indices]

    def call(self, array):
        """Return the genomes reinterpreted"""
        if self.map is not None:
            array = np.logical_xor(self.map, array)
        return array


flipmap = Flipmap()
