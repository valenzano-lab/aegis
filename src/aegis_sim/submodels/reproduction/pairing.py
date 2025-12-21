import numpy as np
from aegis_sim import variables
from aegis_sim.dataclasses.bitarray import BitArray
from aegis_sim import submodels


def get_mating_pairs(parental_sexes):
    """Get mating pairs and return males, females, and pair count."""
    males, females = submodels.matingmanager.pair_up_polygamously(parental_sexes)
    assert len(males) == len(females)
    n_pairs = len(males)
    which_male_gamete = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)
    which_female_gamete = (variables.rng.random(n_pairs) < 0.5).astype(np.int32)
    return males, females, n_pairs, which_male_gamete, which_female_gamete


def pairing(bitarray: BitArray, males, females, n_pairs, which_male_gamete, which_female_gamete):
    """Return assorted chromatids."""

    # Which gamete
    male_genomes = bitarray.get(individuals=males)
    male_gametes = male_genomes[np.arange(n_pairs), which_male_gamete]

    female_genomes = bitarray.get(individuals=females)
    female_gametes = female_genomes[np.arange(n_pairs), which_female_gamete]

    # Unify gametes
    gshape = bitarray.shape()
    children = np.empty(shape=(n_pairs, *gshape[1:]), dtype=np.bool_)
    children[np.arange(n_pairs), 0] = male_gametes
    children[np.arange(n_pairs), 1] = female_gametes

    # TODO fix splitting of ages and muta_prob
    return children
