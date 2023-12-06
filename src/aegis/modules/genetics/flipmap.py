"""Modifier of fitness landscape

Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
"""
import numpy as np
from aegis import pan
from aegis import cnf
from aegis.modules.genetics.gstruc import shape

if cnf.FLIPMAP_CHANGE_RATE == 0:
    dummy = True
else:
    dummy = False
    map_ = np.zeros(shape, dtype=np.bool_)


def call(genomes):
    """Return the genomes reinterpreted"""
    return genomes if dummy else np.logical_xor(map_, genomes)


def evolve():
    """Modify the flipmap"""
    if dummy or (pan.stage % cnf.FLIPMAP_CHANGE_RATE > 0):
        return

    indices = tuple(pan.rng.integers(map_.shape))
    map_[indices] = ~map_[indices]
