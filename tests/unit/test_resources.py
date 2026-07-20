"""Unit tests for the Resources submodel, covering both regrowth rules.

Default rule: an overdrawn pool is clamped to 0, so replenish() returns it to the
full additive increment in a single step -- recovery from depletion is immediate.

RESOURCE_DEFICIT_CARRYOVER: the overdraw carries as a debt which is netted against
the next increment, so a large overshoot leaves the pool near zero and the shortage
persists. This is the rule of Sajina & Valenzano 2016 (arXiv:1602.00723, p.2):

    R(t+1) = (R_t - N_t) * k + Rbar   if leftover > 0
    R(t+1) = (R_t - N_t)     + Rbar   if N_t >= R_t, floored at 0
"""

import numpy as np
import pytest

from aegis_sim.submodels.resources.resources import Resources


def make(add=1000.0, mult=0.0, maximum=None, initial=1000.0, carryover=False):
    r = Resources()
    r.init(
        RESOURCE_ADDITIVE_GROWTH=add,
        RESOURCE_MULTIPLICATIVE_GROWTH=mult,
        RESOURCE_MAXIMUM_AMOUNT=maximum,
        RESOURCE_INITIAL_AMOUNT=initial,
        RESOURCE_DEFICIT_CARRYOVER=carryover,
    )
    return r


class TestScavenge:

    def test_demand_below_capacity_is_met_in_full(self):
        r = make(initial=1000.0)
        got = r.scavenge(np.ones(400))
        assert got.sum() == 400
        assert r.capacity == 600

    def test_demand_above_capacity_is_rationed_equally(self):
        r = make(initial=500.0)
        got = r.scavenge(np.ones(1000))
        # Everyone gets the same share; starvation is not genotype-specific.
        assert np.allclose(got, 0.5)
        assert len(np.unique(got)) == 1


class TestDefaultRule:

    def test_depleted_pool_recovers_to_full_increment_in_one_step(self):
        """The reason crashes stay shallow by default: recovery is immediate."""
        r = make(add=1000.0, initial=500.0)
        r.scavenge(np.ones(5000))       # massive overdraw
        assert r.capacity == 0
        r.replenish()
        assert r.capacity == 1000.0     # full increment, overshoot size irrelevant

    def test_overdraw_size_does_not_affect_recovery(self):
        small, huge = make(initial=500.0), make(initial=500.0)
        small.scavenge(np.ones(600))
        huge.scavenge(np.ones(100_000))
        small.replenish()
        huge.replenish()
        assert small.capacity == huge.capacity == 1000.0

    def test_leftover_regrows_multiplicatively(self):
        r = make(add=1000.0, mult=0.1, initial=1000.0)
        r.scavenge(np.ones(500))        # leftover 500
        r.replenish()
        assert r.capacity == pytest.approx(500 * 1.1 + 1000)

    def test_maximum_caps_the_pool(self):
        r = make(add=1000.0, maximum=1200.0, initial=1000.0)
        r.scavenge(np.ones(100))
        r.replenish()
        assert r.capacity == 1200.0

    def test_none_maximum_means_uncapped(self):
        r = make(add=1000.0, maximum=None, initial=1000.0)
        assert r.RESOURCE_MAXIMUM_AMOUNT == np.inf
        for _ in range(5):
            r.replenish()
        assert r.capacity == 6000.0


class TestDeficitCarryover:

    def test_overdraw_goes_into_debt(self):
        r = make(initial=500.0, carryover=True)
        r.scavenge(np.ones(800))
        assert r.capacity == -300

    def test_debt_is_netted_against_the_increment(self):
        r = make(add=1000.0, initial=500.0, carryover=True)
        r.scavenge(np.ones(800))        # debt of 300
        r.replenish()
        assert r.capacity == 700.0      # 1000 - 300, NOT the full 1000

    def test_large_overshoot_leaves_pool_empty(self):
        """A big enough overshoot wipes the next step's supply -- the deep-crash driver."""
        r = make(add=1000.0, initial=500.0, carryover=True)
        r.scavenge(np.ones(3000))       # debt of 2500, exceeds the increment
        r.replenish()
        assert r.capacity == 0.0        # floored, never negative after replenish

    def test_recovery_scales_with_overshoot_size(self):
        """Unlike the default rule, how hard you overshoot determines recovery."""
        mild, severe = make(initial=500.0, carryover=True), make(initial=500.0, carryover=True)
        mild.scavenge(np.ones(700))     # debt 200
        severe.scavenge(np.ones(1400))  # debt 900
        mild.replenish()
        severe.replenish()
        assert mild.capacity == 800.0
        assert severe.capacity == 100.0
        assert severe.capacity < mild.capacity

    def test_multiplicative_growth_not_applied_to_debt(self):
        """Multiplying a debt would amplify it; the paper applies k only to a surplus."""
        r = make(add=1000.0, mult=0.5, initial=500.0, carryover=True)
        r.scavenge(np.ones(800))        # debt of 300
        r.replenish()
        assert r.capacity == 700.0      # 1000 - 300, not 1000 - 450

    def test_positive_leftover_still_regrows_multiplicatively(self):
        r = make(add=1000.0, mult=0.1, initial=1000.0, carryover=True)
        r.scavenge(np.ones(500))
        r.replenish()
        assert r.capacity == pytest.approx(500 * 1.1 + 1000)


class TestRuleComparison:

    def test_shortage_persists_only_under_carryover(self):
        """Same overshoot, same increment: default recovers fully, carryover does not."""
        default = make(add=1000.0, initial=1000.0, carryover=False)
        carry = make(add=1000.0, initial=1000.0, carryover=True)
        for r in (default, carry):
            r.scavenge(np.ones(2500))   # 1500 beyond the pool
            r.replenish()
        assert default.capacity == 1000.0   # shortage over after one step
        assert carry.capacity == 0.0        # still nothing to eat next step
