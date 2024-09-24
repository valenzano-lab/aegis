"""Wrapper for phenotype vectors."""

import numpy as np
from aegis_sim import parameterization


class Phenotypes:
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

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
