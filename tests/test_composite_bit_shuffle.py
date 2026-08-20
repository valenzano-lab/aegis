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
        probs = architecture.interpreter.call(loci, trait.interpreter, custom_weights=trait.custom_weights)
        expected[:, trait.slice] += probs

    assert np.allclose(result, expected)


CUSTOM_WEIGHTED_CONFIG_PATH = pathlib.Path(__file__).absolute().parent / "test_bit_shuffle_custom_weighted.yml"


def test_custom_weighted_interpreter_reads_logical_bit_order_despite_physical_shuffle():
    """custom_weighted is bit-position-sensitive within a locus (weight[i] applies to
    logical bit i). This confirms the physical bit shuffle does not corrupt that
    mapping -- i.e. compute() correctly un-shuffles before the interpreter runs."""
    aegis_sim.init(custom_config_path=CUSTOM_WEIGHTED_CONFIG_PATH, overwrite=True)
    architecture = submodels.architect.architecture
    from aegis_sim import parameterization

    surv = parameterization.traits["surv"]
    assert surv.interpreter == "custom_weighted"
    assert surv.custom_weights == [10, 3, 1, 1]
    assert architecture.bit_permutation.tolist() != list(range(architecture.length))  # sanity: actually shuffled

    popsize = 4
    genomes = architecture.init_genome_array(popsize)  # (popsize, ploidy, n_loci, BITS_PER_LOCUS), physical order

    # Known logical bit patterns for surv's single locus, individual by individual.
    logical_patterns = np.array(
        [
            [True, False, True, False],  # (10 + 1) / 15
            [False, False, False, False],  # 0 / 15
            [True, True, True, True],  # 15 / 15
            [False, True, False, True],  # (3 + 1) / 15
        ]
    )
    logical = architecture.to_logical(genomes)
    surv_locus = surv.start  # single locus (agespecific=False), same for both chromatids
    logical[:, :, surv_locus, :] = logical_patterns[:, None, :]
    genomes = architecture.to_physical(logical)

    result = architecture.compute(genomes.copy())

    expected_surv = np.array([11 / 15, 0 / 15, 15 / 15, 4 / 15], dtype=np.float32)
    assert np.allclose(result[:, surv_locus], expected_surv)
