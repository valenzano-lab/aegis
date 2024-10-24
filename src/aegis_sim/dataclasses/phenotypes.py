"""Wrapper for phenotype vectors."""

import numpy as np
from aegis_sim import parameterization

from aegis_sim.constants import EVOLVABLE_TRAITS


class Phenotypes:
    def __init__(self, array):

        # TODO perform this check somewhere
        # assert array.shape[1] == constants.TRAIT_N * parameterization.parametermanager.parameters.AGE_LIMIT

        # clip phenotype to [0,1] / Apply lo and hi bound
        array = self.clip_array_to_01(array)
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
        # Traits can be non-evolvable (thus same for all individuals at all ages)

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

    def clip_array_to_01(self, array):

        is_egg_container = len(array.shape) == 1
        if is_egg_container:
            # Eggs do not have computed phenotypes. Only when they hatch, are their phenotypes computed.
            return array

        for trait_name in EVOLVABLE_TRAITS:
            lo = parameterization.traits[trait_name].lo
            hi = parameterization.traits[trait_name].hi

            # get old values
            _, _, slice_ = self.get_trait_position(trait_name=trait_name)
            values = array[:, slice_]

            # clip
            new_values = lo + values * (hi - lo)

            # set new values
            array[:, slice_] = new_values

        return array

    def get_trait_position(self, trait_name):
        index = EVOLVABLE_TRAITS.index(trait_name)
        AGE_LIMIT = parameterization.parametermanager.parameters.AGE_LIMIT
        start = index * AGE_LIMIT
        end = start + AGE_LIMIT
        slice_ = slice(start, end)
        return start, end, slice_

    # @staticmethod
    # def clip_array_to_01(array):
    #     for trait in parameterization.traits.values():
    #         array[:, trait.slice] = Phenotypes.clip_trait_to_01(array[:, trait.slice], trait.name)
    #     return array

    # @staticmethod
    # def clip_trait_to_01(array, traitname):
    #     lo = parameterization.traits[traitname].lo
    #     hi = parameterization.traits[traitname].hi
    #     return lo + array * (hi - lo)

    # # Recording functions
    # def to_feather(self):
    #     for_feather = pd.DataFrame(self.array)
    #     for_feather.columns = for_feather.columns.astype(str)
    #     return for_feather

    # @staticmethod
    # def from_feather(path: pathlib.Path):
    #     from_feather = pd.read_feather(path)
    #     return Phenotypes(array=from_feather)
