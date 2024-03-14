"""This module abstracts away genotype and phenotype logic."""

import numpy as np

from .flipmap import flipmap
from .gstruc import gstruc
from .phenomap import phenomap
from .interpreter import interpreter
from .reproduction.mutation import mutator


def flipmap_evolve(stage):
    flipmap.evolve(stage=stage)


def get_map():
    return phenomap.get_map()


def initialize_genomes(N):
    return gstruc.initialize_genomes(N)


def get_trait(attr):
    return gstruc.get_trait(attr)


def get_number_of_bits():
    return gstruc.get_number_of_bits()


def get_number_of_phenotypic_values():
    return gstruc.get_number_of_phenotypic_values()


def get_phenotypes(genomes):
    """Translate genomes into an array of phenotypes probabilities."""
    # Apply the flipmap
    envgenomes = flipmap.call(genomes.get_array())

    # Apply the interpreter functions
    interpretome = np.zeros(shape=(envgenomes.shape[0], envgenomes.shape[2]), dtype=np.float32)
    for trait in gstruc.get_evolvable_traits():
        loci = envgenomes[:, :, trait.slice]  # fetch
        probs = interpreter.call(loci, trait.interpreter)  # interpret
        interpretome[:, trait.slice] += probs  # add back

    # Apply phenomap
    phenotypes = phenomap.call(interpretome)

    # Apply lo and hi bound
    for trait in gstruc.get_evolvable_traits():
        lo, hi = trait.lo, trait.hi
        phenotypes[:, trait.slice] = phenotypes[:, trait.slice] * (hi - lo) + lo

    return phenotypes


def get_evaluation(population, attr, part=None):
    """
    Get phenotypic values of a certain trait for certain individuals.
    Note that the function returns 0 for individuals that are to not be evaluated.
    """
    which_individuals = np.arange(len(population))
    if part is not None:
        which_individuals = which_individuals[part]

    # first scenario
    trait = gstruc.get_trait(attr)
    if not trait.evolvable:
        probs = trait.initial

    # second and third scenario
    if trait.evolvable:
        which_loci = trait.start
        if trait.agespecific:
            which_loci += population.ages[which_individuals]

        probs = population.phenotypes[which_individuals, which_loci]

    # expand values back into an array with shape of whole population
    final_probs = np.zeros(len(population), dtype=np.float32)
    final_probs[which_individuals] += probs

    return final_probs
