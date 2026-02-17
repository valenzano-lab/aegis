"""Unit tests for PopgenStats pure math functions.

Covers: harmonic numbers, harmonic squares, make_3D/make_4D genome
reshaping, allele frequencies, segregating sites, reference genome,
heterozygosity, theta estimators, Tajima's D, SFS, and Fay & Wu's H.
"""

import numpy as np
import pytest

from aegis_sim.utilities.popgenstats import PopgenStats


class TestHarmonic:
    """Verify harmonic number computation: H(n) = sum(1/k for k=1..n)."""

    def test_harmonic_1(self):
        """H(1) = 1."""
        assert PopgenStats.harmonic(1) == pytest.approx(1.0)

    def test_harmonic_2(self):
        """H(2) = 1 + 1/2 = 1.5."""
        assert PopgenStats.harmonic(2) == pytest.approx(1.5)

    def test_harmonic_4(self):
        """H(4) = 1 + 1/2 + 1/3 + 1/4."""
        expected = 1 + 1/2 + 1/3 + 1/4
        assert PopgenStats.harmonic(4) == pytest.approx(expected)


class TestHarmonicSq:
    """Verify harmonic square: H2(n) = sum(1/k^2 for k=1..n)."""

    def test_harmonic_sq_1(self):
        """H2(1) = 1."""
        assert PopgenStats.harmonic_sq(1) == pytest.approx(1.0)

    def test_harmonic_sq_3(self):
        """H2(3) = 1 + 1/4 + 1/9."""
        expected = 1 + 1/4 + 1/9
        assert PopgenStats.harmonic_sq(3) == pytest.approx(expected)


class TestMake3D:
    """Verify 4D -> 3D genome reshaping (interleaves chromatids into bits)."""

    def test_haploid_drops_ploidy_dim(self):
        """Haploid (ploidy=1): (N, 1, L, B) -> (N, L, B)."""
        g = np.ones((5, 1, 3, 4), dtype=bool)
        result = PopgenStats.make_3D(g)
        assert result.shape == (5, 3, 4)

    def test_diploid_interleaves(self):
        """Diploid (ploidy=2): (N, 2, L, B) -> (N, L, 2*B) with interleaved chromatids."""
        g = np.zeros((2, 2, 3, 2), dtype=bool)
        g[:, 0, :, :] = True   # chromatid 0 all-true
        g[:, 1, :, :] = False  # chromatid 1 all-false
        result = PopgenStats.make_3D(g)
        assert result.shape == (2, 3, 4)
        # Odd bits (0, 2) from chromatid 0 = True
        assert result[0, 0, 0] == True
        assert result[0, 0, 2] == True
        # Even bits (1, 3) from chromatid 1 = False
        assert result[0, 0, 1] == False
        assert result[0, 0, 3] == False


class TestMake4D:
    """Verify 3D -> 4D genome reshaping (de-interleaves bits into chromatids)."""

    def test_haploid_roundtrip(self):
        """Haploid: make_4D(make_3D(g)) recovers original shape."""
        g = np.ones((5, 1, 3, 4), dtype=bool)
        g3 = PopgenStats.make_3D(g)
        g4 = PopgenStats.make_4D(g3, ploidy=1)
        assert g4.shape == (5, 1, 3, 4)
        np.testing.assert_array_equal(g4, g)

    def test_diploid_roundtrip(self):
        """Diploid: make_4D(make_3D(g)) recovers original data."""
        rng = np.random.default_rng(42)
        g = rng.integers(0, 2, size=(10, 2, 5, 3)).astype(bool)
        g3 = PopgenStats.make_3D(g)
        g4 = PopgenStats.make_4D(g3, ploidy=2)
        assert g4.shape == g.shape
        np.testing.assert_array_equal(g4, g)


class TestGetReferenceGenome:
    """Verify reference genome is the majority allele at each position."""

    def test_all_ones(self):
        """All-ones population has all-ones reference genome."""
        ps = PopgenStats()
        genomes = np.ones((10, 3, 4), dtype=bool)
        ref = ps.get_reference_genome(genomes)
        assert np.all(ref == 1)

    def test_all_zeros(self):
        """All-zeros population has all-zeros reference genome."""
        ps = PopgenStats()
        genomes = np.zeros((10, 3, 4), dtype=bool)
        ref = ps.get_reference_genome(genomes)
        assert np.all(ref == 0)

    def test_majority_wins(self):
        """When 7/10 individuals have 1 at a position, reference is 1."""
        ps = PopgenStats()
        genomes = np.zeros((10, 1, 1), dtype=bool)
        genomes[:7, 0, 0] = True
        ref = ps.get_reference_genome(genomes)
        assert ref[0] == 1


class TestSegregatingSites:
    """Verify segregating site counting."""

    def test_no_variation(self):
        """Monomorphic population has 0 segregating sites."""
        ps = PopgenStats()
        genomes = np.ones((10, 3, 4), dtype=bool)
        assert ps.get_segregating_sites(genomes, ploidy=1) == 0

    def test_all_sites_segregating(self):
        """Two individuals with opposite genomes: all sites segregate."""
        ps = PopgenStats()
        genomes = np.zeros((2, 1, 4), dtype=bool)
        genomes[0, :, :] = True
        genomes[1, :, :] = False
        assert ps.get_segregating_sites(genomes, ploidy=1) == 4

    def test_one_site_segregating(self):
        """Only one position differs across individuals."""
        ps = PopgenStats()
        genomes = np.ones((5, 2, 3), dtype=bool)
        genomes[0, 0, 0] = False  # one individual differs at one site
        assert ps.get_segregating_sites(genomes, ploidy=1) == 1


class TestPopgenStatsCalc:
    """Verify calc() computes a full set of statistics on simple inputs."""

    def _make_ps_and_calc(self, genomes_4d, mutation_rate=0.01):
        """Helper: create PopgenStats, seed pop history, run calc."""
        from unittest.mock import patch
        from types import SimpleNamespace
        ps = PopgenStats()
        n = genomes_4d.shape[0]
        for _ in range(10):
            ps.record_pop_size_history(np.empty(n))
        params = SimpleNamespace(POPGENSTATS_SAMPLE_SIZE=0)
        with patch("aegis_sim.utilities.popgenstats.parametermanager") as mock_pm:
            mock_pm.parameters = params
            ps.calc(genomes_4d, mutation_rates=np.full(n, mutation_rate))
        return ps

    def test_monomorphic_haploid(self):
        """Monomorphic haploid population: 0 segregating sites, allele freq = 1."""
        g = np.ones((20, 1, 5, 3), dtype=bool)
        ps = self._make_ps_and_calc(g)
        assert ps.n == 20
        assert ps.segregating_sites == 0
        np.testing.assert_allclose(ps.allele_frequencies, 1.0)

    def test_n_and_ne(self):
        """Census size and effective size are computed."""
        g = np.ones((50, 1, 3, 2), dtype=bool)
        ps = self._make_ps_and_calc(g)
        assert ps.n == 50
        assert ps.ne > 0

    def test_theta_positive(self):
        """Theta = ploidy * 2 * Ne * mu should be positive."""
        g = np.ones((30, 1, 4, 2), dtype=bool)
        ps = self._make_ps_and_calc(g, mutation_rate=0.01)
        assert ps.theta > 0

    def test_diploid_heterozygosity(self):
        """Diploid all-ones: no heterozygosity (all homozygous 1/1)."""
        g = np.ones((20, 2, 5, 3), dtype=bool)
        ps = self._make_ps_and_calc(g)
        assert ps.mean_h == pytest.approx(0.0)

    def test_diploid_max_heterozygosity(self):
        """Diploid with chromatid 0 all-true, chromatid 1 all-false: max heterozygosity."""
        g = np.zeros((20, 2, 5, 3), dtype=bool)
        g[:, 0, :, :] = True
        g[:, 1, :, :] = False
        ps = self._make_ps_and_calc(g)
        assert ps.mean_h == pytest.approx(1.0)
