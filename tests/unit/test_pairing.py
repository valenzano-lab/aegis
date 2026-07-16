"""Unit tests for the pairing module.

Covers:
- Output shape: children genomes have correct dimensions
- Gamete selection: each child gets one chromatid from each parent
- Children genome values come from actual parents (not fabricated)
- Sex balance: number of children = min(males, females)
- All-male or all-female input produces no children
- Ages and muta_prob are returned from the female parent
- Determinism with fixed RNG seed
- Correctness at realistic scale
"""

import numpy as np
import pytest

import aegis_sim.variables as variables
from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim.submodels.reproduction.matingmanager import MatingManager
from aegis_sim.submodels.reproduction.pairing import pairing
from aegis_sim import submodels


@pytest.fixture(autouse=True)
def _init_rng_and_mating():
    """Initialize RNG and mating manager for all tests."""
    variables.rng = np.random.default_rng(42)
    np.random.seed(42)
    submodels.matingmanager = MatingManager()


def _make_genomes(n, ploidy=2, loci=4, bpl=1, value=None):
    """Helper to create Genomes with identifiable per-individual values."""
    if value is not None:
        arr = np.full((n, ploidy, loci, bpl), value, dtype=np.bool_)
    else:
        arr = np.random.default_rng(0).integers(0, 2, size=(n, ploidy, loci, bpl)).astype(np.bool_)
    return Genomes(arr)


class TestPairingOutputShape:
    """Children array has the correct shape."""

    def test_basic_shape(self):
        """4 parents (2M, 2F) → 2 children with same genome dimensions."""
        genomes = _make_genomes(4, ploidy=2, loci=6, bpl=1)
        sexes = np.array([0, 0, 1, 1])  # 2 males, 2 females
        ages = np.arange(4, dtype=np.int32)
        muta_prob = np.ones(4) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)

        assert children.shape == (2, 2, 6, 1)  # 2 children, ploidy 2, 6 loci, 1 bpl

    def test_shape_with_larger_bpl(self):
        """Works with BITS_PER_LOCUS > 1."""
        genomes = _make_genomes(6, ploidy=2, loci=10, bpl=8)
        sexes = np.array([0, 0, 0, 1, 1, 1])
        ages = np.arange(6, dtype=np.int32)
        muta_prob = np.ones(6) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)

        assert children.shape == (3, 2, 10, 8)


class TestPairingSexBalance:
    """Number of children = min(males, females)."""

    def test_equal_sexes(self):
        genomes = _make_genomes(6)
        sexes = np.array([0, 0, 0, 1, 1, 1])
        ages = np.arange(6, dtype=np.int32)
        muta_prob = np.ones(6) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)
        assert len(children) == 3

    def test_more_males(self):
        genomes = _make_genomes(5)
        sexes = np.array([0, 0, 0, 0, 1])
        ages = np.arange(5, dtype=np.int32)
        muta_prob = np.ones(5) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)
        assert len(children) == 1

    def test_more_females(self):
        genomes = _make_genomes(5)
        sexes = np.array([0, 1, 1, 1, 1])
        ages = np.arange(5, dtype=np.int32)
        muta_prob = np.ones(5) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)
        assert len(children) == 1


class TestPairingNoChildren:
    """Edge cases that produce zero children."""

    def test_all_males(self):
        genomes = _make_genomes(3)
        sexes = np.array([0, 0, 0])
        ages = np.arange(3, dtype=np.int32)
        muta_prob = np.ones(3) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)
        assert len(children) == 0

    def test_all_females(self):
        genomes = _make_genomes(3)
        sexes = np.array([1, 1, 1])
        ages = np.arange(3, dtype=np.int32)
        muta_prob = np.ones(3) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)
        assert len(children) == 0


class TestPairingGameteOrigin:
    """Each child's chromatids come from actual parents."""

    def test_children_values_from_parents(self):
        """Give each individual a unique genome; verify children carry parental values."""
        n = 6
        ploidy, loci, bpl = 2, 4, 1
        # Make each individual's genome unique by filling with their index
        arr = np.zeros((n, ploidy, loci, bpl), dtype=np.bool_)
        for i in range(n):
            # Individual i: chromatid 0 all-True if i is even, chromatid 1 all-True if i is odd
            arr[i, 0, :, :] = (i % 2 == 0)
            arr[i, 1, :, :] = (i % 2 == 1)

        genomes = Genomes(arr)
        sexes = np.array([0, 0, 0, 1, 1, 1])
        ages = np.arange(n, dtype=np.int32)
        muta_prob = np.ones(n) * 0.01

        children, _, _, _ = pairing(genomes, sexes, ages, muta_prob)

        # Each child's chromatid 0 should come from a male's gamete
        # Each child's chromatid 1 should come from a female's gamete
        # Both should be valid chromatids from the parent pool
        for c in range(len(children)):
            child_c0 = children[c, 0]  # from male
            child_c1 = children[c, 1]  # from female

            # Must match one of the male chromatids
            male_chromatids = [arr[i, ch] for i in range(3) for ch in range(2)]
            assert any(np.array_equal(child_c0, mc) for mc in male_chromatids), \
                f"Child {c} chromatid 0 doesn't match any male chromatid"

            # Must match one of the female chromatids
            female_chromatids = [arr[i, ch] for i in range(3, 6) for ch in range(2)]
            assert any(np.array_equal(child_c1, fc) for fc in female_chromatids), \
                f"Child {c} chromatid 1 doesn't match any female chromatid"


class TestPairingReturnedMetadata:
    """Ages and muta_prob are returned from the female parent."""

    def test_ages_from_females(self):
        genomes = _make_genomes(4)
        sexes = np.array([0, 0, 1, 1])
        ages = np.array([10, 20, 30, 40], dtype=np.int32)
        muta_prob = np.array([0.01, 0.02, 0.03, 0.04])

        _, returned_ages, returned_muta, _ = pairing(genomes, sexes, ages, muta_prob)

        # Returned ages should be from female indices (2 and 3)
        assert all(a in [30, 40] for a in returned_ages)
        assert len(returned_ages) == 2

    def test_muta_prob_from_females(self):
        genomes = _make_genomes(4)
        sexes = np.array([0, 0, 1, 1])
        ages = np.array([10, 20, 30, 40], dtype=np.int32)
        muta_prob = np.array([0.01, 0.02, 0.03, 0.04])

        _, _, returned_muta, _ = pairing(genomes, sexes, ages, muta_prob)

        assert all(m in [0.03, 0.04] for m in returned_muta)
        assert len(returned_muta) == 2


class TestPairingDeterminism:
    """Same seed produces identical results."""

    def test_deterministic(self):
        genomes = _make_genomes(10, loci=20)
        sexes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ages = np.arange(10, dtype=np.int32)
        muta_prob = np.ones(10) * 0.01

        variables.rng = np.random.default_rng(99)
        np.random.seed(99)
        c1, a1, m1, f1 = pairing(genomes, sexes, ages, muta_prob)

        variables.rng = np.random.default_rng(99)
        np.random.seed(99)
        c2, a2, m2, f2 = pairing(genomes, sexes, ages, muta_prob)

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(f1, f2)


class TestPairingRealisticScale:
    """Correctness at realistic population sizes."""

    def test_large_population(self):
        """1000 parents, 500M 500F, 2000 loci — basic sanity checks."""
        n = 1000
        genomes = _make_genomes(n, ploidy=2, loci=200, bpl=1)
        sexes = np.array([0] * 500 + [1] * 500)
        ages = np.arange(n, dtype=np.int32)
        muta_prob = np.ones(n) * 0.001

        children, returned_ages, returned_muta, returned_females = pairing(genomes, sexes, ages, muta_prob)

        assert len(returned_females) == 500
        assert len(children) == 500
        assert children.shape == (500, 2, 200, 1)
        assert children.dtype == np.bool_
        assert len(returned_ages) == 500
        assert len(returned_muta) == 500
