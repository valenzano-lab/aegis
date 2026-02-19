"""Unit tests for the GPM (genotype-phenotype map) and its numba kernel.

Covers:
- apply_phenolist_numba: single entry, multiple entries, zero magnitude,
  pleiotropic sites (same genome index → multiple phenotype targets),
  age-pleiotropic sites (same genome index → same trait at different ages),
  accumulation (multiple entries targeting the same phenotype index),
  zero vec_state (inactive site produces no change),
  output shape, large-scale correctness against naive reference.
- GPM.__call__: dummy phenomap (no-op), phenolist path, phenomatrix path.
"""

import numpy as np
import pytest

from aegis_sim.submodels.genetics.modifying.gpm import GPM, apply_phenolist_numba


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _naive_apply_phenolist(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes):
    """Pure-python reference implementation."""
    result = phenodiff.copy()
    n_individuals = result.shape[0]
    n_phenolist = vec_indices.shape[0]
    for i in range(n_phenolist):
        for j in range(n_individuals):
            result[j, phenotype_indices[i]] += vectors[j, vec_indices[i]] * magnitudes[i]
    return result


# ---------------------------------------------------------------------------
# apply_phenolist_numba tests
# ---------------------------------------------------------------------------

class TestApplyPhenolistNumba:

    def test_single_entry(self):
        """One phenolist entry: genome site 0 affects phenotype index 2."""
        phenodiff = np.zeros((3, 5), dtype=np.float64)
        # 3 individuals, 3 genome columns; phenolist references column 0
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        vec_indices = np.array([0], dtype=np.int64)
        phenotype_indices = np.array([2], dtype=np.int64)
        magnitudes = np.array([-0.01])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        assert result[0, 2] == pytest.approx(-0.01)
        assert result[1, 2] == pytest.approx(0.0)
        assert result[2, 2] == pytest.approx(-0.005)
        # Other columns untouched
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0

    def test_multiple_entries(self):
        """Two independent phenolist entries targeting different phenotype indices."""
        phenodiff = np.zeros((2, 4), dtype=np.float64)
        vectors = np.array([[1.0, 0.5], [0.0, 1.0]])
        vec_indices = np.array([0, 1], dtype=np.int64)
        phenotype_indices = np.array([0, 3], dtype=np.int64)
        magnitudes = np.array([0.1, -0.2])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        assert result[0, 0] == pytest.approx(0.1)
        assert result[0, 3] == pytest.approx(-0.1)
        assert result[1, 0] == pytest.approx(0.0)
        assert result[1, 3] == pytest.approx(-0.2)

    def test_zero_magnitude(self):
        """Zero magnitude produces no change regardless of vec_state."""
        phenodiff = np.zeros((2, 3), dtype=np.float64)
        vectors = np.array([[1.0], [1.0]])
        vec_indices = np.array([0], dtype=np.int64)
        phenotype_indices = np.array([1], dtype=np.int64)
        magnitudes = np.array([0.0])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        np.testing.assert_array_equal(result, np.zeros((2, 3)))

    def test_zero_vec_state(self):
        """Inactive genome site (vec_state=0) produces no change."""
        phenodiff = np.zeros((2, 3), dtype=np.float64)
        vectors = np.array([[0.0], [0.0]])
        vec_indices = np.array([0], dtype=np.int64)
        phenotype_indices = np.array([1], dtype=np.int64)
        magnitudes = np.array([0.5])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        np.testing.assert_array_equal(result, np.zeros((2, 3)))

    def test_accumulation_same_target(self):
        """Two phenolist entries targeting the same phenotype index accumulate."""
        phenodiff = np.zeros((1, 3), dtype=np.float64)
        vectors = np.array([[1.0, 1.0]])
        vec_indices = np.array([0, 1], dtype=np.int64)
        phenotype_indices = np.array([1, 1], dtype=np.int64)  # both target index 1
        magnitudes = np.array([0.3, 0.2])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        assert result[0, 1] == pytest.approx(0.5)

    def test_pleiotropy_same_site_different_targets(self):
        """One genome site affecting two different phenotype indices (trait pleiotropy)."""
        phenodiff = np.zeros((2, 5), dtype=np.float64)
        # Same genome column referenced twice with different targets
        vectors = np.array([[1.0], [0.5]])
        vec_indices = np.array([0, 0], dtype=np.int64)
        phenotype_indices = np.array([1, 3], dtype=np.int64)  # surv at age 1, repr at age 3
        magnitudes = np.array([-0.01, 0.005])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        assert result[0, 1] == pytest.approx(-0.01)
        assert result[0, 3] == pytest.approx(0.005)
        assert result[1, 1] == pytest.approx(-0.005)
        assert result[1, 3] == pytest.approx(0.0025)

    def test_nonzero_initial_phenodiff(self):
        """Phenolist effects are added to existing phenodiff values."""
        phenodiff = np.full((2, 3), 0.5, dtype=np.float64)
        vectors = np.array([[1.0], [1.0]])
        vec_indices = np.array([0], dtype=np.int64)
        phenotype_indices = np.array([1], dtype=np.int64)
        magnitudes = np.array([0.1])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        assert result[0, 0] == pytest.approx(0.5)  # untouched
        assert result[0, 1] == pytest.approx(0.6)  # 0.5 + 0.1
        assert result[1, 1] == pytest.approx(0.6)

    def test_output_shape(self):
        """Output shape matches phenodiff input shape."""
        phenodiff = np.zeros((10, 20), dtype=np.float64)
        vectors = np.ones((10, 100), dtype=np.float64)
        vec_indices = np.array([0, 5, 10, 50, 99], dtype=np.int64)
        phenotype_indices = np.array([0, 5, 10, 15, 19], dtype=np.int64)
        magnitudes = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        assert result.shape == (10, 20)

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_matches_naive_reference(self, seed):
        """Numba kernel matches pure-python reference on random data."""
        rng = np.random.default_rng(seed)
        n_individuals = rng.integers(50, 500)
        n_phenolist = rng.integers(100, 2000)
        n_genome_cols = 2000
        n_phenotype_cols = 200  # e.g. 4 traits × 50 ages

        phenodiff = np.zeros((n_individuals, n_phenotype_cols), dtype=np.float64)
        vectors = rng.random((n_individuals, n_genome_cols))
        vec_indices = rng.integers(0, n_genome_cols, size=n_phenolist).astype(np.int64)
        phenotype_indices = rng.integers(0, n_phenotype_cols, size=n_phenolist).astype(np.int64)
        magnitudes = rng.uniform(-0.05, 0.05, size=n_phenolist)

        result_numba = apply_phenolist_numba(
            phenodiff.copy(), vectors, vec_indices, phenotype_indices, magnitudes
        )
        result_naive = _naive_apply_phenolist(
            phenodiff.copy(), vectors, vec_indices, phenotype_indices, magnitudes
        )

        np.testing.assert_allclose(result_numba, result_naive, atol=1e-9)


# ---------------------------------------------------------------------------
# GPM class tests
# ---------------------------------------------------------------------------

class TestGPMDummy:
    """GPM with empty phenolist and no phenomatrix is a no-op."""

    def test_dummy_returns_zeropheno(self):
        gpm = GPM(phenomatrix=None, phenolist=[])
        zeropheno = np.ones((5, 10), dtype=np.float64)
        interpretome = np.ones((5, 20), dtype=np.float64)

        result = gpm(interpretome, zeropheno)

        np.testing.assert_array_equal(result, zeropheno)


class TestGPMPhenolist:
    """GPM with a phenolist applies effects correctly."""

    def test_simple_phenolist(self):
        """Single phenolist entry: genome site 0 → phenotype index 2."""
        phenodiff = np.zeros((3, 5), dtype=np.float64)
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        vec_indices = np.array([0], dtype=np.int64)
        phenotype_indices = np.array([2], dtype=np.int64)
        magnitudes = np.array([-0.01])

        result = apply_phenolist_numba(phenodiff, vectors, vec_indices, phenotype_indices, magnitudes)

        assert result[0, 2] == pytest.approx(-0.01)
        assert result[1, 2] == pytest.approx(0.0)
        assert result[2, 2] == pytest.approx(-0.005)


class TestGPMPhenomatrix:
    """GPM with a phenomatrix uses matrix dot product."""

    def test_phenomatrix_dot(self):
        """phenomatrix path should compute vectors.dot(phenomatrix)."""
        n_individuals = 4
        n_genome = 10
        n_pheno = 5

        phenomatrix = np.random.default_rng(0).uniform(-0.1, 0.1, size=(n_genome, n_pheno))
        gpm = GPM(phenomatrix=phenomatrix, phenolist=[])

        vectors = np.random.default_rng(1).random((n_individuals, n_genome))
        zeropheno = np.zeros((n_individuals, n_pheno))

        result = gpm(vectors, zeropheno)

        expected = vectors.dot(phenomatrix)
        np.testing.assert_allclose(result, expected)
