"""Unit tests for the Ploider (diploid-to-haploid conversion).

Covers: init validation (sexual requires ploidy 2), diploid_to_haploid
for homozygous-true, homozygous-false, and heterozygous loci, output
shape, input shape validation, batch processing with mixed zygosity,
symmetry of heterozygous cases, edge dominance factors, output dtype,
and correctness at realistic array sizes.
"""

import numpy as np
import pytest

from aegis_sim.submodels.genetics.ploider import Ploider


def _make_ploider(dominance_factor=0.5):
    p = Ploider()
    p.init(REPRODUCTION_MODE="sexual", DOMINANCE_FACTOR=dominance_factor, PLOIDY=2)
    return p


class TestPloiderInit:
    """Verify init-time validation."""

    def test_sexual_requires_ploidy_2(self):
        """Sexual reproduction with ploidy != 2 raises AssertionError."""
        p = Ploider()
        with pytest.raises(AssertionError):
            p.init(REPRODUCTION_MODE="sexual", DOMINANCE_FACTOR=0.5, PLOIDY=1)


class TestDiploidToHaploid:
    """Verify diploid-to-haploid merging logic.

    Input shape: (individuals, 2 chromatids, loci, bits_per_locus).
    Output shape: (individuals, loci, bits_per_locus).
    Both-true -> 1.0, both-false -> 0.0, heterozygous -> DOMINANCE_FACTOR.
    """

    def test_both_true_gives_one(self):
        """Homozygous 1/1 locus maps to 1.0."""
        p = _make_ploider(dominance_factor=0.5)
        loci = np.ones((1, 2, 1, 1), dtype=bool)
        result = p.diploid_to_haploid(loci)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_both_false_gives_zero(self):
        """Homozygous 0/0 locus maps to 0.0."""
        p = _make_ploider(dominance_factor=0.5)
        loci = np.zeros((1, 2, 1, 1), dtype=bool)
        result = p.diploid_to_haploid(loci)
        assert result[0, 0, 0] == pytest.approx(0.0)

    def test_heterozygous_gives_dominance_factor(self):
        """Heterozygous 1/0 locus maps to DOMINANCE_FACTOR."""
        p = _make_ploider(dominance_factor=0.7)
        loci = np.zeros((1, 2, 1, 1), dtype=bool)
        loci[0, 0, 0, 0] = True
        loci[0, 1, 0, 0] = False
        result = p.diploid_to_haploid(loci)
        assert result[0, 0, 0] == pytest.approx(0.7)

    def test_heterozygous_symmetric(self):
        """0/1 and 1/0 should both map to DOMINANCE_FACTOR."""
        p = _make_ploider(dominance_factor=0.3)
        loci_10 = np.zeros((1, 2, 1, 1), dtype=bool)
        loci_10[0, 0, 0, 0] = True
        loci_01 = np.zeros((1, 2, 1, 1), dtype=bool)
        loci_01[0, 1, 0, 0] = True
        r10 = p.diploid_to_haploid(loci_10)
        r01 = p.diploid_to_haploid(loci_01)
        assert r10[0, 0, 0] == pytest.approx(r01[0, 0, 0])
        assert r10[0, 0, 0] == pytest.approx(0.3)

    def test_output_shape(self):
        """Chromatid dimension is collapsed: (10,2,5,8) -> (10,5,8)."""
        p = _make_ploider()
        loci = np.ones((10, 2, 5, 8), dtype=bool)
        result = p.diploid_to_haploid(loci)
        assert result.shape == (10, 5, 8)

    def test_wrong_ploidy_dim_raises(self):
        """Ploidy dimension != 2 raises AssertionError."""
        p = _make_ploider()
        loci = np.ones((5, 3, 4, 2), dtype=bool)
        with pytest.raises(AssertionError):
            p.diploid_to_haploid(loci)

    def test_wrong_ndim_raises(self):
        """3-D input (missing ploidy dim) raises AssertionError."""
        p = _make_ploider()
        loci = np.ones((5, 2, 4), dtype=bool)
        with pytest.raises(AssertionError):
            p.diploid_to_haploid(loci)

    def test_batch_mixed(self):
        """Batch of three individuals with different zygosity patterns."""
        p = _make_ploider(dominance_factor=0.5)
        loci = np.zeros((3, 2, 1, 1), dtype=bool)
        loci[0, :, 0, 0] = True       # homozygous 1/1
        loci[1, 0, 0, 0] = True       # heterozygous 1/0
        # loci[2] stays all-false       # homozygous 0/0
        result = p.diploid_to_haploid(loci)
        assert result[0, 0, 0] == pytest.approx(1.0)
        assert result[1, 0, 0] == pytest.approx(0.5)
        assert result[2, 0, 0] == pytest.approx(0.0)


class TestDiploidToHaploidDtype:
    """Output dtype must be float (not bool)."""

    def test_output_is_float(self):
        p = _make_ploider()
        loci = np.ones((5, 2, 10, 1), dtype=bool)
        result = p.diploid_to_haploid(loci)
        assert np.issubdtype(result.dtype, np.floating)

    def test_output_is_float32(self):
        """Output should be float32 for memory efficiency."""
        p = _make_ploider()
        loci = np.ones((5, 2, 10, 1), dtype=bool)
        result = p.diploid_to_haploid(loci)
        assert result.dtype == np.float32

    def test_heterozygous_value_is_not_rounded(self):
        """Dominance factor 0.3 must survive as a float, not be truncated."""
        p = _make_ploider(dominance_factor=0.3)
        loci = np.zeros((1, 2, 1, 1), dtype=bool)
        loci[0, 0, 0, 0] = True
        result = p.diploid_to_haploid(loci)
        assert result[0, 0, 0] == pytest.approx(0.3, abs=1e-6)


class TestDiploidToHaploidDominanceEdgeCases:
    """Edge values for DOMINANCE_FACTOR."""

    def test_dominance_zero(self):
        """DOMINANCE_FACTOR=0 means heterozygous is recessive (maps to 0)."""
        p = _make_ploider(dominance_factor=0.0)
        loci = np.zeros((1, 2, 1, 1), dtype=bool)
        loci[0, 0, 0, 0] = True
        result = p.diploid_to_haploid(loci)
        assert result[0, 0, 0] == pytest.approx(0.0)

    def test_dominance_one(self):
        """DOMINANCE_FACTOR=1 means heterozygous is fully dominant (maps to 1)."""
        p = _make_ploider(dominance_factor=1.0)
        loci = np.zeros((1, 2, 1, 1), dtype=bool)
        loci[0, 0, 0, 0] = True
        result = p.diploid_to_haploid(loci)
        assert result[0, 0, 0] == pytest.approx(1.0)


class TestDiploidToHaploidMultipleLoci:
    """Multiple loci per individual with mixed zygosity."""

    def test_multi_loci_mixed(self):
        """Each locus independently resolved."""
        p = _make_ploider(dominance_factor=0.5)
        # 2 individuals, 4 loci, 1 bit per locus
        loci = np.zeros((2, 2, 4, 1), dtype=bool)

        # Individual 0: locus 0 = 1/1, locus 1 = 0/0, locus 2 = 1/0, locus 3 = 0/1
        loci[0, 0, 0, 0] = True; loci[0, 1, 0, 0] = True   # 1/1
        # locus 1 stays 0/0
        loci[0, 0, 2, 0] = True                              # 1/0
        loci[0, 1, 3, 0] = True                              # 0/1

        # Individual 1: all heterozygous 1/0
        loci[1, 0, :, 0] = True
        loci[1, 1, :, 0] = False

        result = p.diploid_to_haploid(loci)

        # Individual 0
        assert result[0, 0, 0] == pytest.approx(1.0)   # 1/1
        assert result[0, 1, 0] == pytest.approx(0.0)   # 0/0
        assert result[0, 2, 0] == pytest.approx(0.5)   # 1/0
        assert result[0, 3, 0] == pytest.approx(0.5)   # 0/1

        # Individual 1: all heterozygous
        for loc in range(4):
            assert result[1, loc, 0] == pytest.approx(0.5)


class TestDiploidToHaploidRealisticSize:
    """Correctness at realistic array sizes (close to actual simulation)."""

    def test_large_array_values_in_range(self):
        """All output values must be in {0.0, DOMINANCE_FACTOR, 1.0}."""
        p = _make_ploider(dominance_factor=0.5)
        rng = np.random.default_rng(42)
        loci = rng.integers(0, 2, size=(1000, 2, 200, 1)).astype(bool)
        result = p.diploid_to_haploid(loci)

        unique_vals = set(np.unique(result))
        assert unique_vals <= {0.0, 0.5, 1.0}

    def test_large_array_homozygous_true_count(self):
        """Count of 1.0 values should match count of both-true loci."""
        p = _make_ploider(dominance_factor=0.5)
        rng = np.random.default_rng(123)
        loci = rng.integers(0, 2, size=(500, 2, 100, 1)).astype(bool)
        result = p.diploid_to_haploid(loci)

        both_true = (loci[:, 0] & loci[:, 1])
        expected_ones = both_true.sum()
        actual_ones = (result == 1.0).sum()
        assert actual_ones == expected_ones

    def test_large_array_homozygous_false_count(self):
        """Count of 0.0 values should match count of both-false loci."""
        p = _make_ploider(dominance_factor=0.5)
        rng = np.random.default_rng(456)
        loci = rng.integers(0, 2, size=(500, 2, 100, 1)).astype(bool)
        result = p.diploid_to_haploid(loci)

        both_false = (~loci[:, 0] & ~loci[:, 1])
        expected_zeros = both_false.sum()
        actual_zeros = (result == 0.0).sum()
        assert actual_zeros == expected_zeros

    def test_large_array_heterozygous_count(self):
        """Count of DOMINANCE_FACTOR values should match count of heterozygous loci."""
        df = 0.5
        p = _make_ploider(dominance_factor=df)
        rng = np.random.default_rng(789)
        loci = rng.integers(0, 2, size=(500, 2, 100, 1)).astype(bool)
        result = p.diploid_to_haploid(loci)

        hetero = (loci[:, 0] ^ loci[:, 1])
        expected_hetero = hetero.sum()
        actual_hetero = (result == df).sum()
        assert actual_hetero == expected_hetero

    @pytest.mark.parametrize("bpl", [1, 8])
    def test_shape_preserved_at_scale(self, bpl):
        """Shape is correct for realistic genome sizes."""
        p = _make_ploider()
        loci = np.ones((2000, 2, 200, bpl), dtype=bool)
        result = p.diploid_to_haploid(loci)
        assert result.shape == (2000, 200, bpl)
