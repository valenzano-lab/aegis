"""Modifier of fitness landscape

Modifies topology of fitness landscape over time by changing the interpretation of zeros and ones in genomes.
"""

import numpy as np
from aegis.hermes import hermes


class Envdrift:
    """
    
    VISOR
    Environmental drift is deactivated when [[ENVDRIFT_RATE]] is 0.
    Conceptually, environmental drift simulates long-term environmental change such as climate change, resource depletion, pollution, etc.
    The main purpose of environmental drift is to allow the population to keep evolving adaptively.
    When the environment does not change, the fitness landscape is static – initially, the population evolves adaptively as it climbs the fitness landscape but once it approaches the fitness peak,
    natural selection acts mostly to purify new detrimental mutations. When environmental drift is activated, the fitness landscape changes over time – thus, the population keeps evolving adaptively, following the fitness peak.

    """
    def __init__(self, ENVDRIFT_RATE, genome_shape):
        self.ENVDRIFT_RATE = ENVDRIFT_RATE

        if self.ENVDRIFT_RATE == 0:
            self.map = None
        else:
            self.map = np.zeros(genome_shape, dtype=np.bool_)

    def evolve(self, step):
        """Modify the envdrift"""
        if (self.map is None) or (step % self.ENVDRIFT_RATE > 0):
            return

        indices = tuple(hermes.rng.integers(self.map.shape))
        self.map[indices] = ~self.map[indices]

    def call(self, array):
        """Return the genomes reinterpreted"""
        if self.map is not None:
            array = np.logical_xor(self.map, array)
        return array
