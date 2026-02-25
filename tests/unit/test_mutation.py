"""Unit tests for the Mutator class.

Covers: init validation (invalid method, mutation ratio rates),
_mutate_by_bit (zero probability, full probability, shape preservation),
and the static apply_mutation_age_multiplier formula.
"""

import numpy as np
import pytest

import aegis_sim.variables as variables
from aegis_sim.submodels.reproduction.mutation import Mutator


@pytest.fixture(autouse=True)
def _init_rng():
    """Ensure variables.rng is initialized for all tests."""
    variables.rng = np.random.default_rng(42)


def _make_mutator(ratio=1.0, method="by_bit", age_multiplier=0.0):
    m = Mutator()
    m.init(MUTATION_RATIO=ratio, MUTATION_METHOD=method, MUTATION_AGE_MULTIPLIER=age_multiplier)
    return m


class TestMutatorInit:
    """Verify init-time validation and rate computation."""

    def test_invalid_method_raises(self):
        """Unknown MUTATION_METHOD raises ValueError."""
        m = Mutator()
        with pytest.raises(ValueError, match="MUTATION_METHOD"):
            m.init(MUTATION_RATIO=1.0, MUTATION_METHOD="invalid", MUTATION_AGE_MULTIPLIER=0.0)

    def test_ratio_one_gives_equal_rates(self):
        """MUTATION_RATIO=1 means 0->1 and 1->0 rates are both 0.5."""
        m = _make_mutator(ratio=1.0)
        assert m.rate_0to1 == pytest.approx(0.5)
        assert m.rate_1to0 == pytest.approx(0.5)

    def test_high_ratio_biases_toward_ones(self):
        """MUTATION_RATIO >> 1 makes 0->1 mutations more likely than 1->0."""
        m = _make_mutator(ratio=10.0)
        assert m.rate_0to1 > m.rate_1to0


class TestMutateByBit:
    """Verify the by-bit mutation method."""

    def test_no_mutation_at_zero_prob(self):
        """Zero mutation probability leaves the genome unchanged."""
        m = _make_mutator(method="by_bit")
        genomes = np.ones((5, 2, 3, 4), dtype=bool)
        original = genomes.copy()
        muta_prob = np.zeros(5, dtype=np.float32)
        ages = np.zeros(5, dtype=np.int32)
        result = m._mutate_by_bit(genomes, muta_prob, ages)
        np.testing.assert_array_equal(result, original)

    def test_full_mutation_flips_bits(self):
        """With probability 1.0 and forced-low random values, all 1-bits flip to 0."""
        m = _make_mutator(ratio=1.0, method="by_bit")
        genomes = np.ones((10, 2, 3, 4), dtype=bool)
        muta_prob = np.ones(10, dtype=np.float32)
        ages = np.zeros(10, dtype=np.int32)
        random_probs = np.full(genomes.shape, 0.001)
        result = m._mutate_by_bit(genomes, muta_prob, ages, random_probabilities=random_probs)
        assert np.all(result == False)

    def test_preserves_shape(self):
        """Output genome array has the same shape as input."""
        m = _make_mutator(method="by_bit")
        shape = (8, 2, 5, 3)
        genomes = np.ones(shape, dtype=bool)
        muta_prob = np.full(8, 0.01, dtype=np.float32)
        ages = np.zeros(8, dtype=np.int32)
        result = m._mutate_by_bit(genomes, muta_prob, ages)
        assert result.shape == shape


class TestApplyMutationAgeMultiplier:
    """Verify the age-dependent mutation rate scaling.

    Formula: final = initial * (1 + age * MUTATION_AGE_MULTIPLIER)
    """

    def test_zero_multiplier_no_change(self):
        """With multiplier=0, mutation probabilities are unchanged."""
        probs = np.array([0.1, 0.2, 0.3])
        ages = np.array([0, 10, 20])
        result = Mutator.apply_mutation_age_multiplier(probs, ages, MUTATION_AGE_MULTIPLIER=0.0)
        np.testing.assert_allclose(result, probs)

    def test_positive_multiplier_increases_with_age(self):
        """Older individuals get higher mutation probabilities.

        age 0:  0.1 * (1 + 0*0.1) = 0.1
        age 5:  0.1 * (1 + 5*0.1) = 0.15
        age 10: 0.1 * (1 + 10*0.1) = 0.2
        """
        probs = np.array([0.1, 0.1, 0.1])
        ages = np.array([0, 5, 10])
        result = Mutator.apply_mutation_age_multiplier(probs, ages, MUTATION_AGE_MULTIPLIER=0.1)
        np.testing.assert_allclose(result, [0.1, 0.15, 0.2])
