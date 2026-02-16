"""Unit tests for the recombination module.

Covers: zero recombination rate (no-op), output shape preservation, and
conservation of total true-bits per individual (recombination swaps
between chromatids but does not create or destroy genetic material).
"""

import numpy as np
import pytest

import aegis_sim.variables as variables
from aegis_sim.submodels.reproduction.recombination import recombination


@pytest.fixture(autouse=True)
def _init_rng():
    """Ensure variables.rng is initialized for all tests."""
    variables.rng = np.random.default_rng(42)


class TestRecombinationZeroRate:
    """Verify that zero recombination rate is a no-op."""

    def test_zero_rate_returns_unchanged(self):
        """RECOMBINATION_RATE=0 returns the input array unmodified."""
        genomes = np.ones((6, 2, 4, 2), dtype=bool)
        result = recombination(genomes, RECOMBINATION_RATE=0)
        np.testing.assert_array_equal(result, genomes)


class TestRecombinationPreservesShape:
    """Verify that recombination does not alter the genome array shape."""

    def test_output_shape_matches_input(self):
        """Output has the same (individuals, ploidy, loci, bits) shape."""
        shape = (10, 2, 5, 3)
        genomes = np.random.default_rng(0).integers(0, 2, size=shape).astype(bool)
        result = recombination(genomes, RECOMBINATION_RATE=0.1)
        assert result.shape == shape


class TestRecombinationPreservesBitCounts:
    """Verify that recombination conserves genetic material per individual."""

    def test_total_true_bits_conserved_approximately(self):
        """Per-individual true-bit count is identical before and after recombination.

        Recombination swaps segments between chromatids within an individual,
        so the total number of 1-bits per individual must be conserved.
        """
        rng = np.random.default_rng(123)
        shape = (100, 2, 10, 4)
        genomes = rng.integers(0, 2, size=shape).astype(bool)
        result = recombination(genomes.copy(), RECOMBINATION_RATE=0.5)
        per_individual_before = genomes.sum(axis=(1, 2, 3))
        per_individual_after = result.sum(axis=(1, 2, 3))
        np.testing.assert_array_equal(per_individual_before, per_individual_after)
