import numpy as np
from numba import njit, prange
from aegis_sim import variables
from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim import submodels


@njit(parallel=True)
def _assemble_children(genome_array, males, females, male_gamete_idx, female_gamete_idx):
    """Assemble children genomes directly from parent genome array.

    Reads each parent's selected chromatid and writes it into the children array
    in one pass, parallelized over pairs. No intermediate arrays allocated.
    """
    n_pairs = len(males)
    # genome_array shape: (n_individuals, ploidy, loci, bpl)
    n_loci = genome_array.shape[2]
    n_bpl = genome_array.shape[3]
    children = np.empty((n_pairs, 2, n_loci, n_bpl), dtype=genome_array.dtype)

    for p in prange(n_pairs):
        m = males[p]
        f = females[p]
        mg = male_gamete_idx[p]
        fg = female_gamete_idx[p]
        for i in range(n_loci):
            for j in range(n_bpl):
                children[p, 0, i, j] = genome_array[m, mg, i, j]
                children[p, 1, i, j] = genome_array[f, fg, i, j]

    return children

@njit(parallel=True)
def _assemble_children_packed(packed_genome_array, males, females, male_gamete_idx, female_gamete_idx):
    """Assemble children genomes directly from packed parent arrays.

    Copies packed bytes from parent chromatids to children in one pass,
    parallelized over pairs. One inner loop over n_packed_bytes instead of
    the two inner loops (n_loci × n_bpl) used by the bool-based kernel.

    Args:
        packed_genome_array: uint8, shape (n_individuals, ploidy, n_packed_bytes)
        males: int array of male parent indices
        females: int array of female parent indices
        male_gamete_idx: int32 array, chromatid selection (0 or 1) per male
        female_gamete_idx: int32 array, chromatid selection (0 or 1) per female

    Returns:
        uint8 array, shape (n_pairs, 2, n_packed_bytes)
    """
    n_pairs = len(males)
    n_packed_bytes = packed_genome_array.shape[2]
    children = np.empty((n_pairs, 2, n_packed_bytes), dtype=np.uint8)

    for p in prange(n_pairs):
        m = males[p]
        f = females[p]
        mg = male_gamete_idx[p]
        fg = female_gamete_idx[p]
        for b in range(n_packed_bytes):
            children[p, 0, b] = packed_genome_array[m, mg, b]
            children[p, 1, b] = packed_genome_array[f, fg, b]

    return children



def pairing(genomes: Genomes, parental_sexes, ages, muta_prob):
    """Return assorted chromatids."""

    # Get pairs
    males, females = submodels.matingmanager.pair_up_polygamously(parental_sexes)
    assert len(males) == len(females)
    n_pairs = len(males)

    if n_pairs == 0:
        gshape = genomes.shape()
        children = np.empty(shape=(0, *gshape[1:]), dtype=np.bool_)
        return children, ages[females], muta_prob[females]

    # Random gamete selection (chromatid 0 or 1 per parent)
    male_gamete_idx = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)
    female_gamete_idx = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)

    # Assemble children directly from genome array — no intermediate copies
    children = _assemble_children(
        genomes.array, males, females, male_gamete_idx, female_gamete_idx,
    )

    # TODO fix splitting of ages and muta_prob
    return children, ages[females], muta_prob[females]

def pairing_packed(packed_genomes, parental_sexes, ages, muta_prob, n_loci, bits_per_locus):
    """Return assorted chromatids from packed uint8 parent genomes.

    Mirrors the logic of pairing() but operates on packed uint8 arrays,
    avoiding the unpack/repack overhead.

    Args:
        packed_genomes: uint8 array, shape (n_individuals, ploidy, n_packed_bytes)
        parental_sexes: int array indicating sex of each parent (0=male, 1=female)
        ages: int array of parent ages
        muta_prob: float array of mutation probabilities per parent
        n_loci: number of loci (unused here, passed through for pipeline consistency)
        bits_per_locus: bits per locus (unused here, passed through for pipeline consistency)

    Returns:
        Tuple of (children_packed, ages, muta_prob) where children_packed is
        uint8 array with shape (n_pairs, 2, n_packed_bytes).
    """

    # Get pairs — same RNG call as pairing()
    males, females = submodels.matingmanager.pair_up_polygamously(parental_sexes)
    assert len(males) == len(females)
    n_pairs = len(males)

    if n_pairs == 0:
        n_packed_bytes = packed_genomes.shape[2]
        children = np.empty(shape=(0, 2, n_packed_bytes), dtype=np.uint8)
        return children, ages[females], muta_prob[females]

    # Random gamete selection — same RNG calls as pairing()
    male_gamete_idx = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)
    female_gamete_idx = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)

    # Assemble children directly from packed genome array
    children = _assemble_children_packed(
        packed_genomes, males, females, male_gamete_idx, female_gamete_idx,
    )

    return children, ages[females], muta_prob[females]

