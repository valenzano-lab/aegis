import numpy as np

from aegis.panconfiguration import pan


class Environment:
    """Modifier of fitness landscape

    Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
    """

    def __init__(self, gstruc_shape, ENVIRONMENT_CHANGE_RATE):

        if ENVIRONMENT_CHANGE_RATE == 0:
            self.dummy = True
        else:
            self.dummy = False
            self.map_ = np.zeros(gstruc_shape, bool)
            self.ENVIRONMENT_CHANGE_RATE = ENVIRONMENT_CHANGE_RATE

    def __call__(self, genomes):
        """Return the genomes reinterpreted in the current environment"""
        return genomes if self.dummy else np.logical_xor(self.map_, genomes)

    def evolve(self):
        """Modify the environmental map"""
        if self.dummy or pan.skip(self.ENVIRONMENT_CHANGE_RATE):
            return

        indices = tuple(pan.rng.integers(self.map_.shape))
        self.map_[indices] = ~self.map_[indices]
