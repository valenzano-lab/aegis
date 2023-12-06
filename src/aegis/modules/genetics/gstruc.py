"""Genome structure

Contains information about ploidy, number of loci, and number of bits per locus.
Calculates phenotypes from input genomes (calls Interpreter, Phenomap and Flipmap).
"""
import numpy as np
from aegis import pan
from aegis import cnf


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
            else:  # one locus for all ages
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


# Generate traits and save
traits = {}
evolvable = []
length = 0

for name in Trait.legal:
    trait = Trait(name, length)
    traits[name] = trait
    length += trait.length
    if trait.evolvable:
        evolvable.append(trait)

# Infer ploidy
ploidy = {
    "sexual": 2,
    "asexual": 1,
    "asexual_diploid": 2,
}[cnf.REPRODUCTION_MODE]


shape = (ploidy, length, cnf.BITS_PER_LOCUS)


def initialize_genomes():
    """Return n initialized genomes.

    Different sections of genome are initialized with a different ratio of ones and zeros
    depending on the G_{}_initial parameter.
    """

    n = cnf.MAX_POPULATION_SIZE
    headsup = cnf.HEADSUP + cnf.MATURATION_AGE if cnf.HEADSUP > -1 else None

    # Initial genomes with a trait.initial fraction of 1's
    genomes = pan.rng.random(size=(n, *shape), dtype=np.float32)

    for trait in evolvable:
        genomes[:, :, trait.slice] = genomes[:, :, trait.slice] <= trait.initial

    genomes = genomes.astype(np.bool_)

    # Guarantee survival and reproduction values up to a certain age
    if headsup is not None:
        surv_start = traits["surv"].start
        repr_start = traits["repr"].start
        genomes[:, :, surv_start : surv_start + headsup] = True
        genomes[:, :, repr_start : repr_start + headsup] = True

    return genomes
