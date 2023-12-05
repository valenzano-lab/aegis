import numpy as np

from aegis.panconfiguration import pan


class Flipmap:
    """Modifier of fitness landscape

    Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
    """

    def __init__(self, gstruc_shape, FLIPMAP_CHANGE_RATE):
        if FLIPMAP_CHANGE_RATE == 0:
            self.dummy = True
        else:
            self.dummy = False
            self.map_ = np.zeros(gstruc_shape, dtype=np.bool_)
            self.FLIPMAP_CHANGE_RATE = FLIPMAP_CHANGE_RATE

    def __call__(self, genomes):
        """Return the genomes reinterpreted"""
        return genomes if self.dummy else np.logical_xor(self.map_, genomes)

    def evolve(self):
        """Modify the flipmap"""
        if self.dummy or pan.skip(self.FLIPMAP_CHANGE_RATE):
            return

        indices = tuple(pan.rng.integers(self.map_.shape))
        self.map_[indices] = ~self.map_[indices]
