import numpy as np


class Population:
    """Population data

    Contains demographic, genetic and phenotypic data of living individuals.
    """

    attrs = (
        "genomes",
        "ages",
        "births",
        "birthdays",
        "phenotypes",
    )

    def __init__(self, genomes, ages, births, birthdays, phenotypes):

        # Main data
        self.genomes = genomes
        self.ages = ages
        self.births = births
        self.birthdays = birthdays
        self.phenotypes = phenotypes

        # Aux data
        self.eggs_ = []
        self.alive_ = np.ones(len(genomes), dtype=np.bool8)
        self.hatch_ = False
        self.n_alive = np.count_nonzero(self.alive_)
        self.n_self = len(genomes)
        self.n_total = len(genomes)

        if not (
            len(genomes)
            == len(ages)
            == len(births)
            == len(birthdays)
            == len(phenotypes)
        ):
            raise ValueError("Population attributes must have equal length")

    def mask(self, mask):
        self.alive_ *= mask
        self.n_alive = np.count_nonzero(self.alive_)

    def add_eggs(self, eggs):
        self.eggs_.append(eggs)
        self.n_total += eggs.n_self

    def reshuffle(self):
        for attr in self.attrs:

            value_self = [
                getattr(self, attr)[self.alive_]
            ]  # Attribute values of surviving individuals from the active population

            value_eggs = [
                getattr(eggs, attr) for eggs in self.eggs_
            ]  # Attribute values of individuals from the egg populations

            value = np.concatenate(value_self + value_eggs)
            setattr(self, attr, value)

        # Reset auxiliary values
        self.hatch_ = False
        self.eggs_ = []

        n = len(self.genomes)
        self.alive_ = np.ones(n, dtype=np.bool8)
        self.n_alive = n
        self.n_self = n
        self.n_total = n
