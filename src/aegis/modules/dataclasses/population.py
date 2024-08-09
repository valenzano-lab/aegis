import numpy as np
import pickle

from aegis.hermes import hermes
from aegis.modules.dataclasses.genomes import Genomes


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
        "infection",
        "sizes",
        "sexes",
    )

    def __init__(self, genomes: Genomes, ages, births, birthdays, phenotypes, infection, sizes, sexes):
        self.genomes = genomes
        self.ages = ages
        self.births = births
        self.birthdays = birthdays
        self.phenotypes = phenotypes
        self.infection = infection
        self.sizes = sizes
        self.sexes = sexes

        if not (
            len(genomes)
            == len(ages)
            == len(births)
            == len(birthdays)
            == len(phenotypes)
            == len(infection)
            == len(sizes)
            == len(sexes)
        ):
            raise ValueError("Population attributes must have equal length")

    def __len__(self):
        """Return the number of living individuals."""
        return len(self.genomes)

    def __getitem__(self, index):
        """Return a subpopulation."""
        return Population(
            genomes=self.genomes.get(individuals=index),
            ages=self.ages[index],
            births=self.births[index],
            birthdays=self.birthdays[index],
            phenotypes=self.phenotypes[index],
            infection=self.infection[index],
            sizes=self.sizes[index],
            sexes=self.sexes[index],
        )

    def __imul__(self, index):
        """Redefine itself as its own subpopulation."""
        for attr in self.attrs:
            if attr == "genomes":
                self.genomes.keep(individuals=index)
            else:
                setattr(self, attr, getattr(self, attr)[index])
        return self

    def __iadd__(self, population):
        """Merge with another population."""

        for attr in self.attrs:
            if attr == "genomes":
                self.genomes.add(population.genomes)
            else:
                val = np.concatenate([getattr(self, attr), getattr(population, attr)])
                setattr(self, attr, val)
        return self

    # def shuffle(self):
    #     order = hermes.rng.arange(len(self))
    #     hermes.rng.shuffle(order)
    #     self *= order

    @staticmethod
    def load_pickle_from(path):
        with open(path, "rb") as file_:
            return pickle.load(file_)

    def save_pickle_to(self, path):
        with open(path, "wb") as file_:
            pickle.dump(self, file_)

    @staticmethod
    def initialize(n):
        genomes = Genomes(hermes.architect.architecture.init_genome_array(n))
        ages = np.zeros(n, dtype=np.int32)
        births = np.zeros(n, dtype=np.int32)
        birthdays = np.zeros(n, dtype=np.int32)
        phenotypes = hermes.architect.__call__(genomes)
        infection = np.zeros(n, dtype=np.int32)
        sizes = np.zeros(n, dtype=np.float32)
        sexes = hermes.modules.sexsystem.get_sex(n)
        return Population(
            genomes=genomes,
            ages=ages,
            births=births,
            birthdays=birthdays,
            phenotypes=phenotypes,
            infection=infection,
            sizes=sizes,
            sexes=sexes,
        )

    @staticmethod
    def make_eggs(offspring_genomes: Genomes, step, offspring_sexes):
        n = len(offspring_genomes)
        eggs = Population(
            genomes=offspring_genomes,
            ages=np.zeros(n, dtype=np.int32),
            births=np.zeros(n, dtype=np.int32),
            birthdays=np.zeros(n, dtype=np.int32) + step,
            # phenotypes=hermes.architect.__call__(offspring_genomes),
            phenotypes=np.empty(n),
            infection=np.zeros(n, dtype=np.int32),
            sizes=np.zeros(n, dtype=np.float32),
            sexes=offspring_sexes,
        )
        return eggs
