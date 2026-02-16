"""Unit tests for the Predation submodel.

Covers: initial predator count, zero-prey edge case, kill fraction
bounds, Verhulst predator growth dynamics, and the effect of
PREDATION_RATE on kill fraction.
"""

import numpy as np
import pytest

from aegis_sim.submodels.predation import Predation


class TestPredationInit:
    """Verify initial state of the Predation model."""

    def test_initial_predator_count(self):
        """Predator population starts at 1."""
        p = Predation(PREDATOR_GROWTH=0.1, PREDATION_RATE=0.05)
        assert p.N == 1


class TestPredationCall:
    """Verify __call__ returns a valid kill fraction and updates predator N."""

    def test_zero_prey_returns_zero(self):
        """No prey means no predation (avoids division by zero)."""
        p = Predation(PREDATOR_GROWTH=0.1, PREDATION_RATE=0.05)
        assert p(prey_count=0) == 0

    def test_returns_fraction(self):
        """Kill fraction is between 0 and 1."""
        p = Predation(PREDATOR_GROWTH=0.1, PREDATION_RATE=0.05)
        frac = p(prey_count=100)
        assert 0 <= frac <= 1

    def test_predator_population_grows(self):
        """Predator N increases after a step with abundant prey."""
        p = Predation(PREDATOR_GROWTH=0.5, PREDATION_RATE=0.05)
        initial_n = p.N
        p(prey_count=1000)
        assert p.N > initial_n

    def test_high_predation_rate_increases_kill_fraction(self):
        """Higher PREDATION_RATE produces a larger kill fraction."""
        p_low = Predation(PREDATOR_GROWTH=0.1, PREDATION_RATE=0.01)
        p_high = Predation(PREDATOR_GROWTH=0.1, PREDATION_RATE=0.5)
        frac_low = p_low(prey_count=100)
        frac_high = p_high(prey_count=100)
        assert frac_high > frac_low

    def test_multiple_steps_predator_approaches_prey(self):
        """Over many steps, predator N converges toward prey count (Verhulst)."""
        p = Predation(PREDATOR_GROWTH=0.3, PREDATION_RATE=0.05)
        for _ in range(50):
            p(prey_count=100)
        assert p.N > 50
