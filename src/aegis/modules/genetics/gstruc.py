"""Genome structure

Contains information about ploidy, number of loci, and number of bits per locus.
"""

import numpy as np
from aegis.pan import rng

from aegis.modules.genetics.trait import Trait


class Gstruc:
    def __init__(self, REPRODUCTION_MODE, BITS_PER_LOCUS):
        # Generate traits and save
        self.traits = {}
        self.evolvable = []
        self.length = 0

        for name in Trait.legal:
            trait = Trait(name, self.length)
            self.traits[name] = trait
            self.length += trait.length
            if trait.evolvable:
                self.evolvable.append(trait)

        # Infer ploidy
        ploidy = {
            "sexual": 2,
            "asexual": 1,
            "asexual_diploid": 2,
        }[REPRODUCTION_MODE]

        self.shape = (ploidy, self.length, BITS_PER_LOCUS)

    def initialize_genomes(self, N):
        """Return n initialized genomes.

        Different sections of genome are initialized with a different ratio of ones and zeros
        depending on the G_{}_initial parameter.
        """

        array = rng.random(size=(N, *self.shape), dtype=np.float32)

        for trait in self.traits.values():
            if trait.evolvable:
                array[:, :, trait.slice] = array[:, :, trait.slice] < trait.initial

        array = array.astype(np.bool_)
        return array

    def get_number_of_bits(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def get_number_of_phenotypic_values(self):
        return self.shape[1]

    def get_trait(self, attr):
        return self.traits[attr]

    def get_shape(self):
        return self.shape

    def get_evolvable_traits(self):
        return self.evolvable
