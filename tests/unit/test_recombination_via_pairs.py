"""Unit tests for recombination_via_pairs and its numba kernel.

Covers:
- Zero recombination rate (no-op)
- Output shape preservation
- Conservation of genetic material (bit counts per individual)
- Correctness of the optimized numba kernel against a naive reference
- Deterministic output with fixed RNG seed
"""

import numpy as np
import pytest

import aegis_sim.variables as variables
from aegis_sim.submodels.reproduction.recombination import (
    recombination_via_pairs,
    recombination_via_pairs_numba,
)


@pytest.fixture(autouse=True)
def _init_rng():
    """Ensure variables.rng is initialized for all tests."""
    variables.rng = np.random.default_rng(42)
    np.random.seed(42)


class TestRecombinationViaPairsZeroRate:
    """Zero recombination rate should be a no-op."""

    def test_zero_rate_returns_unchanged(self):
        genomes = np.ones((10, 2, 8, 1), dtype=np.bool_)
        result = recombination_via_pairs(genomes.copy(), RECOMBINATION_RATE=0)
        np.testing.assert_array_equal(result, genomes)

    def test_zero_rate_various_shapes(self):
        for shape in [(5, 2, 4, 1), (20, 2, 100, 8), (1, 2, 50, 1)]:
            genomes = np.random.default_rng(0).integers(0, 2, size=shape).astype(np.bool_)
            result = recombination_via_pairs(genomes.copy(), RECOMBINATION_RATE=0)
            np.testing.assert_array_equal(result, genomes)


class TestRecombinationViaPairsShape:
    """Output shape must match input shape."""

    @pytest.mark.parametrize("shape", [
        (10, 2, 50, 1),
        (5, 2, 200, 8),
        (50, 2, 10, 1),
    ])
    def test_output_shape_matches_input(self, shape):
        genomes = np.random.default_rng(0).integers(0, 2, size=shape).astype(np.bool_)
        result = recombination_via_pairs(genomes.copy(), RECOMBINATION_RATE=0.01)
        assert result.shape == shape


class TestRecombinationViaPairsBitConservation:
    """Recombination swaps between chromatids — total bits per individual must be conserved."""

    @pytest.mark.parametrize("bpl", [1, 8])
    def test_bit_count_conserved(self, bpl):
        rng = np.random.default_rng(99)
        shape = (100, 2, 50, bpl)
        genomes = rng.integers(0, 2, size=shape).astype(np.bool_)
        result = recombination_via_pairs(genomes.copy(), RECOMBINATION_RATE=0.05)

        before = genomes.sum(axis=(1, 2, 3))
        after = result.sum(axis=(1, 2, 3))
        np.testing.assert_array_equal(before, after)


class TestRecombinationNumbaKernelCorrectness:
    """Verify the optimized numba kernel matches a naive reference implementation."""

    @staticmethod
    def _naive_recombination(flat_genomes, n_recombination_sites, chiasmata_list):
        """Naive reference: apply each chiasma sequentially with slice swaps."""
        for i in range(len(flat_genomes)):
            for j in range(n_recombination_sites[i]):
                chiasma = chiasmata_list[i, j]
                c0 = flat_genomes[i, 0, :chiasma].copy()
                c1 = flat_genomes[i, 1, :chiasma].copy()
                flat_genomes[i, 0, :chiasma] = c1
                flat_genomes[i, 1, :chiasma] = c0
        return flat_genomes

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_matches_naive_reference(self, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(10, 100)
        n_sites = rng.integers(10, 300)
        genomes = rng.integers(0, 2, size=(n, 2, n_sites), dtype=np.bool_)
        rate = 0.01
        n_reco = np.random.default_rng(seed).binomial(n=n_sites, p=rate, size=n)
        max_n = max(max(n_reco), 1)
        chiasmata = rng.integers(1, max(n_sites, 2), size=(n, max_n), dtype=np.int32)

        result_naive = self._naive_recombination(
            genomes.copy(), n_reco, chiasmata.copy()
        )
        result_optimized = recombination_via_pairs_numba(
            genomes.copy(), n_reco, chiasmata.copy()
        )

        np.testing.assert_array_equal(result_naive, result_optimized)

    def test_single_chiasma_swaps_prefix(self):
        """A single chiasma at position k should swap sites [0, k) between chromatids."""
        flat = np.zeros((1, 2, 10), dtype=np.bool_)
        flat[0, 0, :] = True  # chromatid 0 all ones
        flat[0, 1, :] = False  # chromatid 1 all zeros

        n_reco = np.array([1])
        chiasmata = np.array([[5]], dtype=np.int32)

        result = recombination_via_pairs_numba(flat.copy(), n_reco, chiasmata)

        # Sites 0-4 should be swapped, 5-9 unchanged
        expected_c0 = np.array([False]*5 + [True]*5)
        expected_c1 = np.array([True]*5 + [False]*5)
        np.testing.assert_array_equal(result[0, 0], expected_c0)
        np.testing.assert_array_equal(result[0, 1], expected_c1)

    def test_two_chiasmata_cancel_overlap(self):
        """Two identical chiasmata should cancel out (even number of toggles)."""
        flat = np.zeros((1, 2, 10), dtype=np.bool_)
        flat[0, 0, :] = True

        n_reco = np.array([2])
        chiasmata = np.array([[5, 5]], dtype=np.int32)

        result = recombination_via_pairs_numba(flat.copy(), n_reco, chiasmata)

        # Two swaps at same position cancel out — should be unchanged
        np.testing.assert_array_equal(result[0, 0], flat[0, 0])
        np.testing.assert_array_equal(result[0, 1], flat[0, 1])


class TestRecombinationViaPairsDeterminism:
    """Same seed should produce identical results."""

    def test_deterministic_with_same_seed(self):
        shape = (50, 2, 100, 1)
        genomes = np.random.default_rng(0).integers(0, 2, size=shape).astype(np.bool_)

        variables.rng = np.random.default_rng(77)
        np.random.seed(77)
        r1 = recombination_via_pairs(genomes.copy(), RECOMBINATION_RATE=0.01)

        variables.rng = np.random.default_rng(77)
        np.random.seed(77)
        r2 = recombination_via_pairs(genomes.copy(), RECOMBINATION_RATE=0.01)

        np.testing.assert_array_equal(r1, r2)
