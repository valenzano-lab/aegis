import numpy as np
import pickle
import pathlib
import logging
from typing import Optional

from aegis_sim.dataclasses.bitarray import Genomes, Origins
from aegis_sim.dataclasses.phenotypes import Phenotypes
from aegis_sim import submodels


class Population:
    """Population data

    Contains demographic, genetic and phenotypic data of living individuals.
    """

    attrs = (
        "genomes",
        "ages",
        "births",
        "birthdays",
        "generations",
        "phenotypes",
        "infection",
        "sizes",
        "sexes",
        "origins",
    )

    def __init__(
        self,
        genomes: Genomes,
        ages,
        births,
        birthdays,
        phenotypes: Phenotypes,
        infection,
        sizes,
        sexes,
        generations=None,
        origins: Optional[Origins] = None,
    ):
        self.genomes = genomes
        self.ages = ages
        self.births = births
        self.birthdays = birthdays
        self.phenotypes = phenotypes
        self.infection = infection
        self.sizes = sizes
        self.sexes = sexes
        self.generations = generations
        self.origins = origins

        assert isinstance(phenotypes, Phenotypes)
        if origins is not None:
            assert isinstance(origins, Origins), f"Origin is of type {type(origins)}, must be Origins."

        if not (
            len(genomes)
            == len(ages)
            == len(births)
            == len(birthdays)
            == len(phenotypes)
            == len(infection)
            == len(sizes)
            == len(sexes)
            # == len(generations)
        ):
            raise ValueError("Population attributes must have equal length")

    def __len__(self):
        """Return the number of living individuals."""
        return len(self.genomes)

    def __getitem__(self, index):
        """Return a subpopulation."""
        return Population(
            genomes=Genomes(self.genomes.get(individuals=index)),
            ages=self.ages[index],
            births=self.births[index],
            birthdays=self.birthdays[index],
            phenotypes=Phenotypes(self.phenotypes.get(individuals=index)),
            infection=self.infection[index],
            sizes=self.sizes[index],
            sexes=self.sexes[index],
            generations=self.generations[index] if self.generations is not None else None,
            origins=Origins(self.origins.get(individuals=index)) if self.origins is not None else None,
        )

    def __imul__(self, index):
        """Redefine itself as its own subpopulation."""
        for attr in self.attrs:
            if attr == "genomes":
                self.genomes.keep(individuals=index)
            elif attr == "phenotypes":
                self.phenotypes.keep(individuals=index)
            elif attr == "origins":
                if self.origins is not None:
                    self.origins.keep(individuals=index)
            elif attr == "generations":
                self.generations = None
            else:
                setattr(self, attr, getattr(self, attr)[index])
        return self

    def sample(self, fraction):
        """Return a population with a fraction of individuals (at least 1)."""
        if fraction <= 0:
            raise ValueError("Fraction must be greater than 0")
        if fraction > 1:
            raise ValueError("Fraction must be less than or equal to 1")
        
        n_sample = max(1, int(len(self) * fraction))
        logging.info(f"Sampling fraction is {fraction}; {n_sample} individuals from population of {len(self)} sampled.")
        indices = np.random.choice(len(self), size=n_sample, replace=False)
        return self[indices]

    def set_origins(self, origin_tracking_number):
        """Initialize origins if they are not already initialized."""
        self.origins = Origins(submodels.architect.architecture.init_origins_array(
            popsize=len(self), origin_tracking_number=origin_tracking_number))

    def remove_origins(self):
        """Remove origins data. Can be invoked when ORIGIN_TRACKING is 'no_tracking'."""
        self.origins = None

    def __iadd__(self, population):
        """Merge with another population."""

        for attr in self.attrs:
            if attr == "genomes":
                self.genomes.add(population.genomes)
            elif attr == "phenotypes":
                assert isinstance(population.phenotypes, Phenotypes)
                self.phenotypes.add(population.phenotypes)
            elif attr == "origins":
                if self.origins is not None and population.origins is not None:
                    self.origins.add(population.origins)
                elif self.origins is not None and population.origins is None:
                    raise ValueError(f"Cannot merge populations: self has origins ({len(self.origins)} elements) but population to add does not have origins ({len(population)} individuals). Origin tracking must be consistent.")
                elif self.origins is None and population.origins is not None:
                    raise ValueError(f"Cannot merge populations: self does not have origins but population to add has origins ({len(population.origins)} elements). Origin tracking must be consistent.")
            elif attr == "generations":
                self.generations = None
            else:
                val = np.concatenate([getattr(self, attr), getattr(population, attr)])
                setattr(self, attr, val)
        return self

    # def shuffle(self):
    #     order = np.random.arange(len(self))
    #     np.random.shuffle(order)
    #     self *= order

    @staticmethod
    def load_pickle_from(path: pathlib.Path) -> "Population":
        assert path.exists(), f"There is not pickle file at path: {path}. Cannot load population."
        with open(path, "rb") as file_:
            return pickle.load(file_)

    def save_pickle_to(self, path):
        with open(path, "wb") as file_:
            pickle.dump(self, file_)

    @staticmethod
    def initialize(n, AGE_LIMIT) -> "Population":
        genomes = Genomes(submodels.architect.architecture.init_genome_array(n))
        ages = np.random.randint(low=0, high=AGE_LIMIT, size=n, dtype=np.int32)
        births = np.zeros(n, dtype=np.int32)
        birthdays = np.zeros(n, dtype=np.int32)
        # generations = np.zeros(n, dtype=np.int32)

        phenotypes = submodels.architect.__call__(genomes)
        assert isinstance(phenotypes, Phenotypes)

        infection = np.zeros(n, dtype=np.int32)
        sizes = np.zeros(n, dtype=np.float32)
        sexes = submodels.sexsystem.get_sex(n)
        return Population(
            genomes=genomes,
            ages=ages,
            births=births,
            birthdays=birthdays,
            generations=None,
            phenotypes=phenotypes,
            infection=infection,
            sizes=sizes,
            sexes=sexes,
            origins=None,
        )

    @staticmethod
    def make_eggs(offspring_genomes: Genomes, step, offspring_sexes, parental_generations, offspring_origins=None) -> "Population":
        n = len(offspring_genomes)
        eggs = Population(
            genomes=offspring_genomes,
            ages=np.zeros(n, dtype=np.int32),
            births=np.zeros(n, dtype=np.int32),
            birthdays=np.zeros(n, dtype=np.int32) + step,
            # generations=parental_generations + 1,
            generations=None,
            # phenotypes=submodels.architect.__call__(offspring_genomes), # Do not compute phenotypes until eggs are laid! Why? Because it is computationally expensive.
            phenotypes=Phenotypes.init_phenotype_array(n),
            infection=np.zeros(n, dtype=np.int32),
            sizes=np.zeros(n, dtype=np.float32),
            sexes=offspring_sexes,
            origins=offspring_origins,
        )
        return eggs
