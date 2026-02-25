"""Unit tests for the Population dataclass.

Covers: constructor validation, __len__, __getitem__ (documents a known
bug), __imul__ (in-place filtering), __iadd__ (merging), pickle
round-trip, and the make_eggs static factory.
"""

import numpy as np
import pytest

from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim.dataclasses.phenotypes import Phenotypes
from aegis_sim.dataclasses.population import Population


def _make_phenotypes(n, length=5):
    """Create a Phenotypes instance bypassing clip logic."""
    p = Phenotypes.__new__(Phenotypes)
    p.array = np.random.rand(n, length).astype(np.float32)
    return p


def _make_pop(n=10, phenotype_len=5):
    """Create a minimal Population for testing (no submodel dependencies)."""
    genomes = Genomes(np.ones((n, 2, 3), dtype=bool))
    ages = np.arange(n, dtype=np.int32)
    births = np.zeros(n, dtype=np.int32)
    birthdays = np.zeros(n, dtype=np.int32)
    infection = np.zeros(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.float32)
    sexes = np.zeros(n, dtype=np.int32)
    phenotypes = _make_phenotypes(n, phenotype_len)
    return Population(
        genomes=genomes,
        ages=ages,
        births=births,
        birthdays=birthdays,
        phenotypes=phenotypes,
        infection=infection,
        sizes=sizes,
        sexes=sexes,
    )


class TestPopulationInit:
    """Verify that the constructor rejects mismatched attribute lengths."""

    def test_length_mismatch_raises(self):
        """Passing arrays of different lengths raises ValueError."""
        genomes = Genomes(np.ones((3, 2), dtype=bool))
        ages = np.zeros(4, dtype=np.int32)  # wrong length
        births = np.zeros(3, dtype=np.int32)
        birthdays = np.zeros(3, dtype=np.int32)
        infection = np.zeros(3, dtype=np.int32)
        sizes = np.zeros(3, dtype=np.float32)
        sexes = np.zeros(3, dtype=np.int32)
        pheno = Phenotypes.__new__(Phenotypes)
        pheno.array = np.zeros((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="equal length"):
            Population(
                genomes=genomes, ages=ages, births=births,
                birthdays=birthdays, phenotypes=pheno,
                infection=infection, sizes=sizes, sexes=sexes,
            )


class TestPopulationLen:
    """Verify __len__ returns the number of individuals."""

    def test_len(self):
        """Population of 7 reports length 7."""
        pop = _make_pop(7)
        assert len(pop) == 7

    def test_len_empty(self):
        """Empty population reports length 0."""
        pop = _make_pop(0)
        assert len(pop) == 0


class TestPopulationGetitem:
    """Verify __getitem__ slicing.

    NOTE: All tests here are xfail because __getitem__ passes raw ndarrays
    from Genomes.get() / Phenotypes.get() into Population.__init__, which
    asserts isinstance(phenotypes, Phenotypes). This is a known bug.
    """

    @pytest.mark.xfail(
        reason="Bug: Population.__getitem__ passes raw arrays from Genomes.get()/Phenotypes.get() "
               "to Population.__init__ which asserts isinstance(phenotypes, Phenotypes)",
        strict=True,
    )
    def test_slice(self):
        """Slicing pop[2:5] should return a 3-individual subpopulation."""
        pop = _make_pop(10)
        sub = pop[2:5]
        assert len(sub) == 3
        assert isinstance(sub, Population)

    @pytest.mark.xfail(reason="Same __getitem__ bug as test_slice", strict=True)
    def test_index_array(self):
        """Integer index array should select specific individuals."""
        pop = _make_pop(10)
        sub = pop[np.array([0, 3, 7])]
        assert len(sub) == 3

    @pytest.mark.xfail(reason="Same __getitem__ bug as test_slice", strict=True)
    def test_bool_mask(self):
        """Boolean mask should select True-flagged individuals."""
        pop = _make_pop(5)
        mask = np.array([True, False, True, False, True])
        sub = pop[mask]
        assert len(sub) == 3

    @pytest.mark.xfail(reason="Same __getitem__ bug as test_slice", strict=True)
    def test_preserves_ages(self):
        """Sliced subpopulation should carry the correct age values."""
        pop = _make_pop(5)
        sub = pop[np.array([2, 4])]
        np.testing.assert_array_equal(sub.ages, np.array([2, 4], dtype=np.int32))


class TestPopulationImul:
    """Verify __imul__ filters the population in place."""

    def test_imul_filters_in_place(self):
        """pop *= index_array keeps only the selected individuals."""
        pop = _make_pop(10)
        keep = np.array([0, 5, 9])
        pop *= keep
        assert len(pop) == 3

    def test_imul_with_bool_mask(self):
        """pop *= bool_mask keeps only True-flagged individuals."""
        pop = _make_pop(4)
        mask = np.array([True, False, False, True])
        pop *= mask
        assert len(pop) == 2


class TestPopulationIadd:
    """Verify __iadd__ merges two populations."""

    def test_iadd_merges(self):
        """pop1 += pop2 results in combined length."""
        pop1 = _make_pop(5)
        pop2 = _make_pop(3)
        pop1 += pop2
        assert len(pop1) == 8

    def test_iadd_preserves_data(self):
        """Merged population retains attribute values from both sources."""
        pop1 = _make_pop(2)
        pop2 = _make_pop(2)
        pop2.ages = np.array([99, 100], dtype=np.int32)
        pop1 += pop2
        assert pop1.ages[-1] == 100


class TestPopulationPickle:
    """Verify pickle-based save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        """Population survives a pickle round-trip with data intact."""
        pop = _make_pop(5)
        path = tmp_path / "pop.pkl"
        pop.save_pickle_to(path)
        loaded = Population.load_pickle_from(path)
        assert len(loaded) == 5
        np.testing.assert_array_equal(loaded.ages, pop.ages)

    def test_load_missing_raises(self, tmp_path):
        """Loading from a nonexistent path raises AssertionError."""
        import pathlib
        with pytest.raises(AssertionError):
            Population.load_pickle_from(tmp_path / "nonexistent.pkl")


class TestPopulationMakeEggs:
    """Verify the make_eggs static factory.

    Skipped because it requires the full parameterization system (traits
    dict) to be initialized. Covered by functional tests instead.
    """

    @pytest.mark.skip(reason="Requires full parameterization init (traits dict); covered by functional tests")
    def test_make_eggs_shape(self):
        """Eggs population has correct length, zero ages, and correct birthdays."""
        import aegis_sim.parameterization as parameterization
        parameterization.expected_phenotype_length = 10
        n = 5
        genomes = Genomes(np.ones((n, 2, 3), dtype=bool))
        sexes = np.zeros(n, dtype=np.int32)
        eggs = Population.make_eggs(genomes, step=10, offspring_sexes=sexes, parental_generations=None)
        assert len(eggs) == n
        assert np.all(eggs.ages == 0)
        assert np.all(eggs.birthdays == 10)
