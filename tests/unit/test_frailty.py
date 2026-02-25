"""Unit tests for the Frailty submodel.

Covers: the age-dependent hazard modifier formula
hazard * (1 + (age / AGE_LIMIT) * FRAILTY_MODIFIER), including edge
cases (zero modifier, youngest age, oldest age) and vectorized input.
"""

import numpy as np
import pytest

from aegis_sim.submodels.frailty import Frailty


def _make_frailty(modifier=1.0, age_limit=50):
    f = Frailty()
    f.init(FRAILTY_MODIFIER=modifier, AGE_LIMIT=age_limit)
    return f


class TestFrailtyModify:
    """Verify the modify() hazard scaling formula."""

    def test_zero_modifier_no_change(self):
        """With FRAILTY_MODIFIER=0, hazard is unchanged at any age."""
        f = _make_frailty(modifier=0.0)
        ages = np.array([0, 10, 25, 49])
        result = f.modify(hazard=0.1, ages=ages)
        np.testing.assert_allclose(result, 0.1)

    def test_youngest_unaffected(self):
        """Age 0 contributes zero frailty regardless of modifier."""
        f = _make_frailty(modifier=2.0, age_limit=50)
        ages = np.array([0])
        result = f.modify(hazard=0.1, ages=ages)
        assert result[0] == pytest.approx(0.1)

    def test_oldest_gets_full_modifier(self):
        """At age == AGE_LIMIT, the full modifier is applied.

        hazard * (1 + (50/50) * 2.0) = 0.1 * 3.0 = 0.3
        """
        f = _make_frailty(modifier=2.0, age_limit=50)
        ages = np.array([50])
        result = f.modify(hazard=0.1, ages=ages)
        assert result[0] == pytest.approx(0.3)

    def test_mid_age_proportional(self):
        """At half the age limit, half the modifier is applied.

        0.2 * (1 + 0.5 * 1.0) = 0.2 * 1.5 = 0.3
        """
        f = _make_frailty(modifier=1.0, age_limit=100)
        ages = np.array([50])
        result = f.modify(hazard=0.2, ages=ages)
        assert result[0] == pytest.approx(0.3)

    def test_vectorized(self):
        """modify() works on arrays of ages, returning per-individual hazards."""
        f = _make_frailty(modifier=1.0, age_limit=100)
        ages = np.array([0, 50, 100])
        result = f.modify(hazard=0.1, ages=ages)
        expected = np.array([0.1, 0.15, 0.2])
        np.testing.assert_allclose(result, expected)
