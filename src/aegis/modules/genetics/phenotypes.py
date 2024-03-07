"""Wrapper for phenotype vectors."""

import numpy as np
from .flipmap import flipmap
from .gstruc import gstruc
from .phenomap import phenomap
from .interpreter import interpreter


class Phenotyper:
    """
    Order of elements in the vector does not matter.
    """

    def __init__(self, MAX_LIFESPAN):
        self.MAX_LIFESPAN = MAX_LIFESPAN

    def genome_to_phenotype(self, genomes):
        # Environmental drift
        envgenomes = flipmap.call(genomes.get_array())

        # Collapse loci into a vector of [0,1] values
        interpretome = np.zeros(shape=(envgenomes.shape[0], envgenomes.shape[2]), dtype=np.float32)
        for trait in gstruc.get_evolvable_traits():
            loci = envgenomes[:, :, trait.slice]  # fetch
            probs = interpreter.call(loci, trait.interpreter)  # interpret
            interpretome[:, trait.slice] += probs  # add back

        # Phenomapping
        phenotypes = phenomap.call(interpretome)

        # Apply lo and hi bound
        for trait in gstruc.get_evolvable_traits():
            lo, hi = trait.lo, trait.hi
            phenotypes[:, trait.slice] = phenotypes[:, trait.slice] * (hi - lo) + lo

        return phenotypes

    def _phenomap(self, vectors, phenomatrix=None, phenolist=None):
        """
        Vectors .. matrix of shape (individual, ?); items are real numbers, chromosomes and loci are resolved
        Phenomatrix ..
        Phenolist .. list of (bit_index, trait, age, magnitude)
        """
        if phenomatrix is not None:
            # TODO BUG resolve phenomatrix
            return vectors.dot(phenomatrix)
        elif phenolist is not None:
            popsize = len(vectors)
            phenodiff = np.zeros(shape=(popsize, self.MAX_LIFESPAN * 4))
            for vec_index, trait, age, magnitude in phenolist:
                vec_state = vectors[:, vec_index]
                phenotype_change = vec_state * magnitude
                phenotype_index = self.__where(trait, age)
                phenodiff[:, phenotype_index] += phenotype_change
            return phenodiff
        else:
            raise Exception("Neither phenomatrix nor phenolist has been provided.")

    def __where(self, trait, age):
        # Used for phenolist
        # Order of traits is hard-encoded and is: surv, repr, muta, neut
        order = {"surv": 0, "repr": 1, "muta": 2, "neut": 3}
        return self.MAX_LIFESPAN * order[trait] + age
