import numpy as np
import pathlib
import pickle

from aegis_sim.dataclasses.phenotypes import Phenotypes
from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim import submodels
from aegis_sim import variables

from aegis_sim.parameterization import parametermanager
from aegis_sim import parameterization


class Group:
    def __init__(self, individuals):
        assert isinstance(individuals, list)
        self.individuals = individuals

    def remove_individual(self, individual):
        self.individuals.remove(individual)

    def add_individual(self, individual):
        self.individuals.append(individual)

    def __len__(self):
        return len(self.individuals)

    def __iadd__(self, group):
        self.individuals.extend(group.individuals)
        return self

    def __imul__(self, index):
        new_individuals = []
        for condition, individual in zip(index, self.individuals):
            if condition:
                new_individuals.append(individual)
        self.individuals = new_individuals
        return self

    def __getitem__(self, index):
        return Group(self.individuals[index])

    @staticmethod
    def make_eggs(offspring_genomes: Genomes, step, offspring_sexes, parental_generations):
        n = len(offspring_genomes)

        eggs = []

        for i in range(n):

            egg = Individual(
                genome=offspring_genomes[i],
                phenotype=None,
                age=0,
                births=0,
                birthday=step,
                generation=None,
                infection=0,
                size=0,
                sex=offspring_sexes[i],
            )
            eggs.append(egg)

        group = Group(eggs)

        return group

    @staticmethod
    def initialize(n, AGE_LIMIT):
        individuals = []

        for _ in range(n):
            age = variables.rng.integers(low=0, high=AGE_LIMIT, size=1, dtype=np.int32)[0]
            sex = submodels.sexsystem.get_sex(1)[0]
            genome = Genomes(submodels.architect.architecture.init_genome_array(1))
            phenotype = submodels.architect.__call__(genome)
            assert isinstance(phenotype, Phenotypes)

            individual = Individual(
                genome=genome,
                age=age,
                births=0,
                birthday=0,
                phenotype=phenotype,
                infection=0,
                size=0,
                sex=sex,
                generation=None,
            )

            individuals.append(individual)

        return Group(individuals=individuals)

    @staticmethod
    def load_pickle_from(path: pathlib.Path):
        assert path.exists(), f"pickle_path {path} does not exist"
        with open(path, "rb") as file_:
            return pickle.load(file_)

    def save_pickle_to(self, path):
        with open(path, "wb") as file_:
            pickle.dump(self, file_)

    @property
    def ages(self):
        return np.array([individual.age for individual in self.individuals])

    @property
    def births(self):
        return np.array([individual.births for individual in self.individuals])

    @property
    def birthdays(self):
        return np.array([individual.birthday for individual in self.individuals])

    @property
    def sizes(self):
        return np.array([individual.size for individual in self.individuals])

    @property
    def sexes(self):
        return np.array([individual.sex for individual in self.individuals])

    @property
    def infection(self):
        return np.array([individual.infection for individual in self.individuals])

    @property
    def genomes(self):
        arrays = np.array(
            [
                (individual.genome.array[0] if len(individual.genome.array.shape) == 4 else individual.genome.array)
                for individual in self.individuals
            ]
        )
        genomes = Genomes(arrays)
        return genomes

    @property
    def phenotypes(self):
        arrays = np.array(
            [
                (
                    individual.phenotype.array[0]
                    if len(individual.phenotype.array.shape) == 2
                    else individual.phenotype.array
                )
                for individual in self.individuals
            ]
        )
        return Phenotypes(arrays)

    def increment_age(self):
        for individual in self.individuals:
            individual.increment_age()

    def increase_sizes(self):
        for individual in self.individuals:
            individual.increase_size(amount=1)

    def increase_births(self, amounts):
        for individual, amount in zip(self.individuals, amounts):
            individual.increase_births(amount=amount)

    def set_phenotypes(self, phenotypes):
        for individual, phenotype in zip(self.individuals, phenotypes.array):
            individual.set_phenotype(phenotype)

    def extract_traits(self, trait_name, part=None):
        values = []

        if part is None:
            for individual in self.individuals:
                value = individual.get_trait(trait_name)
                values.append(value)

        else:
            for individual, condition in zip(self.individuals, part):
                if not condition:
                    value = 0
                else:
                    value = individual.get_trait(trait_name)
                values.append(value)

        values = np.array(values)
        return values


class Individual:
    def __init__(
        self,
        genome,
        age,
        births,
        birthday,
        phenotype,
        infection,
        size,
        sex,
        generation,
    ):
        self.genome = genome
        self.age = age
        self.births = births
        self.birthday = birthday
        self.phenotype = phenotype
        self.infection = infection
        self.size = size
        self.sex = sex
        self.generation = generation

        if self.phenotype is not None:
            self.unpack_phenotype()

    def increment_age(self):
        self.age += 1

    def increase_size(self, amount):
        self.size += 1

    def increase_births(self, amount):
        self.births += amount

    def set_phenotype(self, phenotype):
        self.phenotype = Phenotypes(phenotype)
        self.unpack_phenotype()

    def unpack_phenotype(self):
        if len(self.phenotype.array.shape) == 2:
            array = self.phenotype.array[0]
        else:
            array = self.phenotype.array

        assert len(array.shape) == 1

        self.phenotype.array = array

    #     s, e, slice_ = Phenotypes.get_trait_position(trait_name="surv")
    #     self.trait_surv = array[s:e]
    #     s, e, slice_ = Phenotypes.get_trait_position(trait_name="repr")
    #     self.trait_repr = array[s:e]
    #     s, e, slice_ = Phenotypes.get_trait_position(trait_name="muta")
    #     self.trait_muta = array[s:e]
    #     print(s, e, slice_)
    #     s, e, slice_ = Phenotypes.get_trait_position(trait_name="neut")
    #     self.trait_neut = array[s:e]

    def get_trait(self, trait_name):

        trait = parameterization.traits[trait_name]

        if not trait.evolvable:
            return trait.initpheno
        else:
            position = trait.start
            if trait.agespecific:
                position += self.age
            return self.phenotype.array[position]

    def __str__(self):
        return self.birthday

    # def get(self, individuals=slice(None), loci=slice(None)):
    #     return self.array[individuals, loci]

    #     # TODO Shift some responsibilities to Phenotypes dataclass

    #     # Generate index of target individuals

    #     n_individuals = len(self)

    #     which_individuals = np.arange(n_individuals)

    #     if part is not None:
    #         which_individuals = which_individuals[part]

    #     # Fetch trait in question
    #     trait = parameterization.traits[trait_name]

    #     # Reminder.
    #     # Traits can be evolvable and age-specific (thus different across individuals and ages)
    #     # Traits can be evolvable and non age-specific (thus different between individuals but same across ages)
    #     # Traits can be non-evolvable (thus same for all individuals at all ages)

    #     if not trait.evolvable:
    #         probs = trait.initpheno
    #     else:
    #         which_loci = trait.start
    #         if trait.agespecific:
    #             shift_by_age = ages[which_individuals]
    #             which_loci += shift_by_age
    #         probs = self.get(which_individuals, which_loci)

    #     # expand values back into an array with shape of whole population
    #     final_probs = np.zeros(n_individuals, dtype=np.float32)
    #     final_probs[which_individuals] += probs

    #     return final_probs
