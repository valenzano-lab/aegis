"""Wrapper for phenotype vectors."""

import numpy as np
import pandas as pd
from aegis_sim import parameterization
from aegis_sim import constants
import pathlib


class Phenotypes:
    def __init__(self, array):

        # TODO perform this check somewhere
        # assert array.shape[1] == constants.TRAIT_N * parameterization.parametermanager.parameters.AGE_LIMIT

        # clip phenotype to [0,1] / Apply lo and hi bound
        # TODO ??? extract slicing
        array = Phenotypes.clip_array_to_01(array)

        self.array = array

    def __len__(self):
        return len(self.array)

    # def to_frame(self, AGE_LIMIT):

    #     df = pd.DataFrame(self.array)

    #     # Edit index
    #     # df.reset_index(drop=True, inplace=True)
    #     # df.index.name = "individual"

    #     # Edit columns
    #     traits = constants.EVOLVABLE_TRAITS

    #     # AGE_LIMIT = parameterization.parametermanager.parameters.AGE_LIMIT
    #     ages = [str(a) for a in range(AGE_LIMIT)]
    #     multi_columns = pd.MultiIndex.from_product([traits, ages], names=["trait", "age"])
    #     df.columns = multi_columns

    #     return df

    def get(self, individuals=slice(None), loci=slice(None)):
        return self.array[individuals, loci]

    def add(self, phenotypes):
        self.array = np.concatenate([self.array, phenotypes.array])

    def keep(self, individuals):
        self.array = self.array[individuals]

    def extract(self, trait_name, ages, part=None):

        # TODO Shift some responsibilities to Phenotypes dataclass

        # Generate index of target individuals

        n_individuals = len(self)

        which_individuals = np.arange(n_individuals)

        if part is not None:
            which_individuals = which_individuals[part]

        # Fetch trait in question
        trait = parameterization.traits[trait_name]

        # Reminder.
        # Traits can be evolvable and age-specific (thus different across individuals and ages)
        # Traits can be evolvable and non age-specific (thus different between individuals but same across ages)
        # Traits can be not evolvable (thus same for all individuals at all ages)

        if not trait.evolvable:
            probs = trait.initpheno
        else:
            which_loci = trait.start
            if trait.agespecific:
                shift_by_age = ages[which_individuals]
                which_loci += shift_by_age
            probs = self.get(which_individuals, which_loci)

        # expand values back into an array with shape of whole population
        final_probs = np.zeros(n_individuals, dtype=np.float32)
        final_probs[which_individuals] += probs

        return final_probs

    @staticmethod
    def clip_array_to_01(array):
        for trait in parameterization.traits.values():
            array[:, trait.slice] = Phenotypes.clip_trait_to_01(array[:, trait.slice], trait.name)
        return array

    @staticmethod
    def clip_trait_to_01(array, traitname):
        lo = parameterization.traits[traitname].lo
        hi = parameterization.traits[traitname].hi
        return lo + array * (hi - lo)

    # # Recording functions
    # def to_feather(self):
    #     for_feather = pd.DataFrame(self.array)
    #     for_feather.columns = for_feather.columns.astype(str)
    #     return for_feather

    # @staticmethod
    # def from_feather(path: pathlib.Path):
    #     from_feather = pd.read_feather(path)
    #     return Phenotypes(array=from_feather)
