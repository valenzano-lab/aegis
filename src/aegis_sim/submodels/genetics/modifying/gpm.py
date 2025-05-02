import logging
import numpy as np
from aegis_sim import parameterization
from numba import njit


class GPM:
    """Genotype-phenotype map

    Order of elements in the vector does not matter. # TODO Explain better

    ### GENOTYPE-PHENOTYPE MAP (GPM) ###
    In AEGIS, every individual carries a genome which encodes an intrinsic phenotype.
    A genome can be converted into an intrinsic phenotype using the genotype-phenotype map (GPM).
    Conceptually, the GPM contains the information on how each site affects the intrinsic phenotype
    of the individual (e.g. the first site decreases fertility by 0.15% at age class 28).

    The GPM can be saved in two formats: a list or a matrix.

    If it is a list, it will be a list of quadruple (4-tuple) with the following structure: `index`, `trait`, `age`, `magnitude`.
    Thus a single quadruple encodes an effect of a single site at the index `index` (e.g. 1)
    on the trait `trait` (e.g. fertility) expressed at the age `age` (e.g. 28). The change to the trait is of magnitude `magnitude` (0.85).
    When a site is pleiotropic, there will be multiple quadruples with the same `index`.
    We distringuish between age-pleiotropy (a single site affecting at least one trait at multiple ages) and trait-pleiotropy (a single site affecting multiple traits).

    If the GPM is encoded as a matrix, it is a 3D matrix where dimensions encode `index`, `trait` and `age`,
    while the matrix values encode the `magnitude`s.

    When most sites are age-pleiotropic and trait-pleiotropic, the optimal encoding format is a matrix.
    When most sites have age-specific and trait-specific effects, the optimal encoding format is a list
    rather than a matrix because the matrix will be very sparse (it will carry a lot of 0's).
    """

    def __init__(self, phenomatrix, phenolist):
        self.phenomatrix = phenomatrix
        self.phenolist = phenolist

        self.dummy = len(self.phenolist) == 0 and self.phenomatrix is None
        if self.dummy:
            logging.info("Phenomap inactive.")

    def phenodiff(self, vectors, zeropheno):
        """
        vectors .. haploidized genomes of all individuals; shape is (n_individuals, ?)
        phenomatrix ..
        phenolist .. list of (bit_index, trait, age, magnitude)
        zeropheno .. phenotypes of all-zero genomes (i.e. how phenotypes would be if all bits were 0)
        """

        if self.phenomatrix is not None:
            # TODO BUG resolve phenomatrix
            return vectors.dot(self.phenomatrix)

        elif self.phenolist is not None:
            phenodiff = zeropheno.copy()
            for vec_index, trait, age, magnitude in self.phenolist:
                vec_state = vectors[:, vec_index]
                phenotype_change = vec_state * magnitude
                phenotype_index = parameterization.traits[trait].start + age
                phenodiff[:, phenotype_index] += phenotype_change
            return phenodiff

        else:
            raise Exception("Neither phenomatrix nor phenolist has been provided.")

    def phenodiff_accelerated(self, vectors, zeropheno):
        if self.phenomatrix is not None:
            return vectors.dot(self.phenomatrix)

        elif self.phenolist is not None:
            phenodiff = zeropheno.copy()
            vec_indices, traits, ages, magnitudes = zip(*self.phenolist)
            vec_indices = np.array(vec_indices)
            magnitudes = np.array(magnitudes)
            vec_states = vectors[:, vec_indices]
            phenotype_indices = np.array(
                [parameterization.traits[trait].start + age for trait, age in zip(traits, ages)]
            )
            phenodiff = apply_phenolist_numba(phenodiff, vec_states, phenotype_indices, magnitudes)
            return phenodiff

        else:
            raise Exception("Neither phenomatrix nor phenolist has been provided.")

    def __call__(self, interpretome, zeropheno):
        if self.dummy:
            return zeropheno
        else:
            # phenodiff_old = self.phenodiff(vectors=interpretome, zeropheno=zeropheno)
            phenodiff = self.phenodiff_accelerated(vectors=interpretome, zeropheno=zeropheno)
            return phenodiff


@njit
def apply_phenolist_numba(phenodiff, vec_states, phenotype_indices, magnitudes):
    """
    phenodiff .. modification to phenotype vector for a specific trait; shape is (individual, phenotypic site)
    vec_states .. state of each genomic site for each individual; shape is (individual, genomic site)
    phenotype_indices .. position in the phenotype vector that is modified by the genomic site; shape is (genomic site)
    magnitudes .. effect size for each site; shape is (genomic site)
    """
    n_individuals, n_phenos = vec_states.shape
    for i in range(n_phenos):
        for j in range(n_individuals):
            phenodiff[j, phenotype_indices[i]] += vec_states[j, i] * magnitudes[i]
    return phenodiff
