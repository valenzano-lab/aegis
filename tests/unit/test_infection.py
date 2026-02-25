"""Unit tests for the Infection submodel.

Covers: the get_infection_probability() formula
(BACKGROUND_INFECTIVITY - 0.5 + logistic(TRANSMISSIBILITY * density)),
including zero density, high density, and zero transmissibility cases.
"""

import math
import pytest

from aegis_sim.submodels.infection import Infection


class TestInfectionProbability:
    """Verify the infection probability formula under different conditions."""

    def test_zero_density(self):
        """At zero infection density, probability equals BACKGROUND_INFECTIVITY.

        Formula: 0.1 - 0.5 + 1/(1+exp(0)) = 0.1 - 0.5 + 0.5 = 0.1
        """
        inf = Infection(
            BACKGROUND_INFECTIVITY=0.1,
            TRANSMISSIBILITY=1.0,
            RECOVERY_RATE=0.05,
            FATALITY_RATE=0.01,
        )
        prob = inf.get_infection_probability(infection_density=0.0)
        assert prob == pytest.approx(0.1)

    def test_high_density_increases_probability(self):
        """Higher infection density drives the logistic term up, raising probability."""
        inf = Infection(
            BACKGROUND_INFECTIVITY=0.1,
            TRANSMISSIBILITY=5.0,
            RECOVERY_RATE=0.05,
            FATALITY_RATE=0.01,
        )
        prob_low = inf.get_infection_probability(0.0)
        prob_high = inf.get_infection_probability(1.0)
        assert prob_high > prob_low

    def test_zero_transmissibility(self):
        """With TRANSMISSIBILITY=0, density has no effect; probability equals BACKGROUND_INFECTIVITY."""
        inf = Infection(
            BACKGROUND_INFECTIVITY=0.2,
            TRANSMISSIBILITY=0.0,
            RECOVERY_RATE=0.05,
            FATALITY_RATE=0.01,
        )
        prob_a = inf.get_infection_probability(0.0)
        prob_b = inf.get_infection_probability(1.0)
        assert prob_a == pytest.approx(prob_b)
        assert prob_a == pytest.approx(0.2)
