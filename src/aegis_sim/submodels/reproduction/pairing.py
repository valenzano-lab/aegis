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


def pairing(genomes: Genomes, parental_sexes, ages, muta_prob, ancestry=None,
            parent_positions=None, max_search_radius=0):
    """Return assorted chromatids.

    When `parent_positions` is given (LATTICE_MODE), the mating manager does
    lattice-aware expanding-ring pairing. Females who can't find a male within
    `max_search_radius` rings are not paired. Otherwise (default), classical
    well-mixed pairing happens.

    Returns `(children, child_ages, child_muta_prob, [child_ancestry], female_slots)`
    where `female_slots` are the slot indices (into the reproducing pool) of the
    paired mothers — the caller uses these to look up mother positions when
    placing offspring on the lattice.
    """

    # Get pairs
    males, females = submodels.matingmanager.pair_up_polygamously(
        parental_sexes,
        parent_positions=parent_positions,
        max_search_radius=max_search_radius,
    )
    assert len(males) == len(females)
    n_pairs = len(males)

    if n_pairs == 0:
        gshape = genomes.shape()
        children = np.empty(shape=(0, *gshape[1:]), dtype=np.bool_)
        empty_female_slots = np.empty(0, dtype=np.int64)
        if ancestry is not None:
            empty_ancestry = np.empty(shape=(0, *ancestry.shape[1:]), dtype=ancestry.dtype)
            return children, ages[females], muta_prob[females], empty_ancestry, empty_female_slots
        return children, ages[females], muta_prob[females], empty_female_slots

    # Random gamete selection (chromatid 0 or 1 per parent)
    male_gamete_idx = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)
    female_gamete_idx = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)

    # Assemble children directly from genome array — no intermediate copies
    children = _assemble_children(
        genomes.array, males, females, male_gamete_idx, female_gamete_idx,
    )

    if ancestry is not None:
        offspring_ancestry = np.empty((n_pairs, *ancestry.shape[1:]), dtype=ancestry.dtype)
        offspring_ancestry[:, 0] = ancestry[males, male_gamete_idx]
        offspring_ancestry[:, 1] = ancestry[females, female_gamete_idx]
        # TODO fix splitting of ages and muta_prob
        return children, ages[females], muta_prob[females], offspring_ancestry, females

    # TODO fix splitting of ages and muta_prob
    return children, ages[females], muta_prob[females], females
