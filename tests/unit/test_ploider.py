"""Unit tests for the Ploider (diploid-to-haploid conversion).

Covers: init validation (sexual requires ploidy 2), diploid_to_haploid
for homozygous-true, homozygous-false, and heterozygous loci, output
shape, input shape validation, and batch processing with mixed zygosity.
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
