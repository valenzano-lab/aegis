import numpy as np
from aegis.hermes import hermes
from aegis.modules.dataclasses.genomes import Genomes


def pairing(genomes: Genomes, parental_sexes, ages, muta_prob):
    """Return assorted chromatids."""

    # Get pairs
    males, females = hermes.modules.matingmanager.pair_up_polygamously(parental_sexes)
    assert len(males) == len(females)
    n_pairs = len(males)

    # Which gamete
    male_genomes = genomes.get(individuals=males)
    which_gamete = (hermes.rng.random(n_pairs) < 0.5).astype(np.int32)
    male_gametes = male_genomes[np.arange(n_pairs), which_gamete]

    female_genomes = genomes.get(individuals=females)
    which_gamete = (hermes.rng.random(n_pairs) < 0.5).astype(np.int32)
    female_gametes = female_genomes[np.arange(n_pairs), which_gamete]

    # Unify gametes
    gshape = genomes.shape()
    children = np.empty(shape=(n_pairs, *gshape[1:]), dtype=np.bool_)
    children[np.arange(n_pairs), 0] = male_gametes
    children[np.arange(n_pairs), 1] = female_gametes

    # TODO fix splitting of ages and muta_prob
    return children, ages[females], muta_prob[females]