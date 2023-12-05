"""Modifier of fitness landscape

Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
"""
import numpy as np

from aegis import pan
from aegis.help import other


FLIPMAP_CHANGE_RATE = None
dummy = None
map_ = None


def init(self, gstruc_shape):
    if FLIPMAP_CHANGE_RATE == 0:
        self.dummy = True
    else:
        self.dummy = False
        self.map_ = np.zeros(gstruc_shape, dtype=np.bool_)


def do(genomes):
    """Return the genomes reinterpreted"""
    return genomes if dummy else np.logical_xor(map_, genomes)


def evolve():
    """Modify the flipmap"""
    if dummy or (pan.stage % FLIPMAP_CHANGE_RATE > 0):
        return

    indices = tuple(other.rng.integers(map_.shape))
    map_[indices] = ~map_[indices]
