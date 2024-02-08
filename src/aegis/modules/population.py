import numpy as np
import pickle


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
    )

    def __init__(self, genomes, ages, births, birthdays, phenotypes, infection):
        self.genomes = genomes
        self.ages = ages
        self.births = births
        self.birthdays = birthdays
        self.phenotypes = phenotypes
        self.infection = infection

        if not (len(genomes) == len(ages) == len(births) == len(birthdays) == len(phenotypes) == len(infection)):
            raise ValueError("Population attributes must have equal length")

    def __len__(self):
        """Return the number of living individuals."""
        return len(self.genomes)

    def __getitem__(self, index):
        """Return a subpopulation."""
        return Population(
            genomes=self.genomes[index],
            ages=self.ages[index],
            births=self.births[index],
            birthdays=self.birthdays[index],
            phenotypes=self.phenotypes[index],
            infection=self.infection[index],
        )

    def __imul__(self, index):
        """Redefine itself as its own subpopulation."""
        for attr in self.attrs:
            setattr(self, attr, getattr(self, attr)[index])
        return self

    def __iadd__(self, population):
        """Merge with another population."""
        for attr in self.attrs:
            val = np.concatenate([getattr(self, attr), getattr(population, attr)])
            setattr(self, attr, val)
        return self

    @staticmethod
    def load_pickle_from(path):
        with open(path, "rb") as file_:
            return pickle.load(file_)

    def save_pickle_to(self, path):
        with open(path, "wb") as file_:
            pickle.dump(self, file_)
