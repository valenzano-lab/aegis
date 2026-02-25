import logging
import numpy as np
from aegis_sim import parameterization
from numba import njit, prange


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

        # Pre-resolved arrays for phenodiff_accelerated (cached to avoid
        # recomputing every call). Populated lazily on first use because
        # parameterization.traits may not be ready at __init__ time.
        self._resolved = False
        self._vec_indices = None
        self._phenotype_indices = None
        self._magnitudes = None

    def _resolve_phenolist(self):
        """Pre-resolve phenolist into numpy arrays (once)."""
        if self._resolved:
            return
        vec_indices, traits, ages, magnitudes = zip(*self.phenolist)
        self._vec_indices = np.array(vec_indices, dtype=np.int64)
        self._magnitudes = np.array(magnitudes, dtype=np.float64)
        self._phenotype_indices = np.array(
            [parameterization.traits[trait].start + age for trait, age in zip(traits, ages)],
            dtype=np.int64,
        )
        self._resolved = True

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
            self._resolve_phenolist()
            phenodiff = zeropheno.copy()
            phenodiff = apply_phenolist_numba(
                phenodiff, vectors, self._vec_indices, self._phenotype_indices, self._magnitudes,
            )
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


@njit(parallel=True)
def apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes):
    """Apply phenolist effects to phenodiff, parallelized over individuals.

    Instead of pre-gathering vec_states = vectors[:, vec_indices] (which allocates
    a large temporary array), this kernel indexes directly into vectors.
    The outer loop is over individuals (prange) so each thread writes to its own
    row of phenodiff — no race conditions.

    Args:
        phenodiff: (n_individuals, n_phenotype_cols) output array, modified in place
        vectors: (n_individuals, genome_size) haploidized genome values
        vec_indices: (n_phenolist,) genome column index for each phenolist entry
        phenotype_indices: (n_phenolist,) phenotype column index for each entry
        magnitudes: (n_phenolist,) effect size for each entry
    """
    n_individuals = phenodiff.shape[0]
    n_phenolist = vec_indices.shape[0]
    for j in prange(n_individuals):
        for i in range(n_phenolist):
            phenodiff[j, phenotype_indices[i]] += vectors[j, vec_indices[i]] * magnitudes[i]
    return phenodiff
