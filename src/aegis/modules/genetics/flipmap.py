"""Modifier of fitness landscape

Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
"""

import numpy as np
from aegis.pan import var
from aegis.pan import cnf
from aegis.modules.genetics.gstruc import gstruc

if cnf.FLIPMAP_CHANGE_RATE == 0:
    dummy = True
else:
    dummy = False
    map_ = np.zeros(gstruc.get_shape(), dtype=np.bool_)


def call(genomes):
    """Return the genomes reinterpreted"""
    return genomes if dummy else np.logical_xor(map_, genomes)


def evolve(FLIPMAP_CHANGE_RATE=cnf.FLIPMAP_CHANGE_RATE):
    """Modify the flipmap"""
    if dummy or (var.stage % FLIPMAP_CHANGE_RATE > 0):
        return

    indices = tuple(var.rng.integers(map_.shape))
    map_[indices] = ~map_[indices]
