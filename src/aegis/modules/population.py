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
        "lineage",
    )

    def __init__(self, genomes, ages, births, birthdays, phenotypes, lineage):
        self.genomes = genomes
        self.ages = ages
        self.births = births
        self.birthdays = birthdays
        self.phenotypes = phenotypes
        self.lineage = lineage

        self.lineage_genomes = genomes.copy()

        if not (
            len(genomes)
            == len(ages)
            == len(births)
            == len(birthdays)
            == len(phenotypes)
            == len(lineage)
        ):
            raise ValueError("Population attributes must have equal length")

    def reset_lineage(self):
        """Restart lineage documentation by assigning each individual a new, unique lineage identifier."""
        self.lineage = np.arange(len(self), dtype=np.int32)
        self.lineage_genomes = self.genomes.copy()
        assert self.lineage_genomes is not self.genomes

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
            lineage=self.lineage[index],
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
