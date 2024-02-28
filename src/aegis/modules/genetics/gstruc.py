"""Genome structure

Contains information about ploidy, number of loci, and number of bits per locus.
"""

import numpy as np
from aegis.pan import cnf
from aegis.pan import var

from aegis.modules.genetics.genomes import Genomes


class Trait:
    """Genetic trait

    Contains data on traits encoded in the genome.
    """

    legal = ("surv", "repr", "neut", "muta")

    def __init__(self, name, start):
        def get(key):
            return getattr(cnf, f"G_{name}_{key}")

        self.name = name

        # Attributes set by the configuration files
        self.evolvable = get("evolvable")
        self.agespecific = get("agespecific")
        self.interpreter = get("interpreter")
        self.lo = get("lo")
        self.hi = get("hi")
        self.initial = get("initial")

        # Determine the number of loci encoding the trait
        if self.evolvable:
            if self.agespecific is True:  # one locus per age
                self.length = cnf.MAX_LIFESPAN
            elif self.agespecific is False:  # one locus for all ages
                self.length = 1
            else:  # custom number of loci
                self.length = self.agespecific
        else:  # no loci for a constant trait
            self.length = 0

        self._validate()

        # Infer positions in the genome
        self.start = start
        self.end = self.start + self.length
        self.slice = slice(self.start, self.end)

    def _validate(self):
        """Check whether input parameters are legal."""
        if not isinstance(self.evolvable, bool):
            raise TypeError

        if not 0 <= self.initial <= 1:
            raise ValueError

        if self.evolvable:
            # if not isinstance(self.agespecific, bool):
            #     raise TypeError

            if self.interpreter not in (
                "uniform",
                "exp",
                "binary",
                "binary_exp",
                "binary_switch",
                "switch",
                "linear",
                "single_bit",
                "const1",
                "threshold",
            ):
                raise ValueError(f"{self.interpreter} is not a valid interpreter type")

            if not 0 <= self.lo <= 1:
                raise ValueError

            if not 0 <= self.hi <= 1:
                raise ValueError

    def __len__(self):
        """Return number of loci used to encode the trait."""
        return self.length

    def __str__(self):
        return self.name


class Gstruc:
    def init(self):
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
        }[cnf.REPRODUCTION_MODE]

        self.shape = (ploidy, self.length, cnf.BITS_PER_LOCUS)

    def initialize_genomes(self, N):
        """Return n initialized genomes.

        Different sections of genome are initialized with a different ratio of ones and zeros
        depending on the G_{}_initial parameter.
        """
        
        genomes = Genomes(var.rng.random(size=(N, *self.shape), dtype=np.float32))
        return genomes

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


gstruc = Gstruc()
