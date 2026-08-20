import pathlib

import numpy as np
import pytest

import aegis_sim
from aegis_sim import submodels

CONFIG_PATH = pathlib.Path(__file__).absolute().parent / "test_bit_shuffle.yml"


@pytest.fixture(scope="module")
def architecture():
    aegis_sim.init(custom_config_path=CONFIG_PATH, overwrite=True)
    return submodels.architect.architecture


def test_bit_permutation_is_a_valid_permutation(architecture):
    perm = architecture.bit_permutation
    assert perm.shape == (architecture.length,)
    assert sorted(perm.tolist()) == list(range(architecture.length))


def test_bit_permutation_is_not_trivial(architecture):
    """With BITS_PER_LOCUS > 1 and several loci, an identity permutation would be
    a near-impossible coincidence; this guards against a no-op shuffle."""
    perm = architecture.bit_permutation
    assert not np.array_equal(perm, np.arange(architecture.length))


def test_bit_permutation_breaks_locus_grouping(architecture):
    """Bits that belong to the same logical locus should not, in general, all land
    on physically adjacent storage positions -- otherwise this would be no better
    than the locus-level shuffle and linkage within a locus would remain intact."""
    bpl = architecture.BITS_PER_LOCUS
    if bpl < 2:
        pytest.skip("Need BITS_PER_LOCUS > 1 to test within-locus bit dispersal")

    perm = architecture.bit_permutation
    n_loci = architecture.n_loci
    all_contiguous = True
    for locus in range(n_loci):
        logical_bits = perm[locus * bpl : (locus + 1) * bpl]
        if not np.array_equal(np.sort(logical_bits), np.arange(logical_bits.min(), logical_bits.min() + bpl)):
            all_contiguous = False
            break
    assert not all_contiguous


def test_to_logical_to_physical_roundtrip(architecture):
    rng = np.random.default_rng(1)
    array = rng.random(size=(5, 2, architecture.n_loci, architecture.BITS_PER_LOCUS)) < 0.5

    physical = architecture.to_physical(array)
    assert physical.shape == array.shape

    back = architecture.to_logical(physical)
    assert np.array_equal(back, array)


def test_init_genome_array_respects_trait_initgeno_after_unshuffle(architecture):
    from aegis_sim import parameterization

    popsize = 20000
    array = architecture.init_genome_array(popsize)  # physical order
    logical = architecture.to_logical(array)  # (popsize, ploidy, n_loci, BITS_PER_LOCUS)

    for trait in parameterization.traits.values():
        if trait.length == 0:
            continue
        bits = logical[:, :, trait.slice]
        observed_rate = bits.mean()
        assert observed_rate == pytest.approx(trait.initgeno, abs=0.02)


def test_compute_matches_manual_unshuffled_interpretation(architecture):
    from aegis_sim import parameterization

    popsize = 50
    genomes = architecture.init_genome_array(popsize)  # (popsize, ploidy, n_loci, BITS_PER_LOCUS), physical order

    result = architecture.compute(genomes.copy())

    # Manually reproduce compute()'s logic to confirm physical storage is
    # correctly un-shuffled before interpretation.
    y = genomes.shape[1]
    if y == 1:
        collapsed = genomes[:, 0]
    else:
        from aegis_sim.submodels.genetics import ploider

        collapsed = ploider.ploider.diploid_to_haploid(genomes)

    logical = architecture.to_logical(collapsed)

    expected = np.zeros(shape=(logical.shape[0], logical.shape[1]), dtype=np.float32)
    for trait in parameterization.traits.values():
        loci = logical[:, trait.slice]
        probs = architecture.interpreter.call(loci, trait.interpreter)
        expected[:, trait.slice] += probs

    assert np.allclose(result, expected)
