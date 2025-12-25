import numpy as np
from aegis_sim import variables
from aegis_sim.dataclasses.bitarray import BitArray
from aegis_sim import submodels


def get_mating_pairs(parental_sexes):
    """Return two arrays of indices for males and females which are paired so that i-th male is paired with i-th female."""
    males, females = submodels.matingmanager.pair_up_polygamously(parental_sexes)
    assert len(males) == len(females)
    return males, females


def get_which_gametes(n_pairs):
    which_male_gamete = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)
    which_female_gamete = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)
    return which_male_gamete, which_female_gamete


def pair(bitarray: BitArray, males, females, which_male_gamete, which_female_gamete):
    """Return assorted chromatids."""

    n_pairs = len(males)

    # Which gamete
    male_bitarrays = bitarray.get(individuals=males)
    male_gametes = male_bitarrays[np.arange(n_pairs), which_male_gamete]

    female_bitarrays = bitarray.get(individuals=females)
    female_gametes = female_bitarrays[np.arange(n_pairs), which_female_gamete]

    # Unify gametes
    gshape = bitarray.shape()
    children = np.empty(shape=(n_pairs, *gshape[1:]), dtype=bitarray.dtype)
    children[np.arange(n_pairs), 0] = male_gametes
    children[np.arange(n_pairs), 1] = female_gametes

    # TODO fix splitting of ages and muta_prob
    return children
