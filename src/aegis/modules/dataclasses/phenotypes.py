"""Wrapper for phenotype vectors."""

import numpy as np


class Phenotypes:
    def __init__(self, array):
        self.array = array

    # @staticmethod
    # def zeros(popsize, number_of_phenotypic_values):
    #     return np.zeros(shape=(popsize, number_of_phenotypic_values))

    # @staticmethod
    # def where(trait, age, AGE_LIMIT):
    #     # Used for phenolist
    #     # Order of traits is hard-encoded and is: surv, repr, muta, neut
    #     order = {"surv": 0, "repr": 1, "muta": 2, "neut": 3}
    #     return AGE_LIMIT * order[trait] + age

    # def get(self, trait, age, AGE_LIMIT):
    #     return self.array[:, self.where(trait, age, AGE_LIMIT)]
