"""Unit tests for the Envdrift (environmental drift) module.

Covers: will_evolve (rate gating), evolve (map bit flipping),
call (XOR reinterpretation), and disabled state (rate=0).
"""

import numpy as np
import pytest

from aegis_sim.submodels.genetics.envdrift import Envdrift


class TestEnvdriftInit:
    """Verify initialization with active and inactive rates."""

    def test_zero_rate_no_map(self):
        """ENVDRIFT_RATE=0 means map is None (disabled)."""
        ed = Envdrift(ENVDRIFT_RATE=0, genome_shape=(2, 5, 3))
        assert ed.map is None

    def test_nonzero_rate_creates_map(self):
        """Nonzero rate creates a zero-filled bool map."""
        ed = Envdrift(ENVDRIFT_RATE=10, genome_shape=(2, 5, 3))
        assert ed.map is not None
        assert ed.map.shape == (2, 5, 3)
        assert ed.map.dtype == np.bool_
        assert np.all(ed.map == False)


class TestWillEvolve:
    """Verify will_evolve step gating logic."""

    def test_disabled_never_evolves(self):
        """Rate=0 -> will_evolve always False."""
        ed = Envdrift(ENVDRIFT_RATE=0, genome_shape=(2, 3))
        assert ed.will_evolve(step=10) is False

    def test_evolves_at_rate_multiple(self):
        """Step divisible by rate -> True."""
        ed = Envdrift(ENVDRIFT_RATE=10, genome_shape=(2, 3))
        assert ed.will_evolve(step=10) is True
        assert ed.will_evolve(step=20) is True

    def test_no_evolve_off_rate(self):
        """Step not divisible by rate -> False."""
        ed = Envdrift(ENVDRIFT_RATE=10, genome_shape=(2, 3))
        assert ed.will_evolve(step=7) is False
        assert ed.will_evolve(step=15) is False


class TestEvolve:
    """Verify evolve flips exactly one bit in the map per call."""

    def test_evolve_changes_map(self):
        """After evolving at a valid step, the map has exactly one True bit."""
        np.random.seed(42)
        ed = Envdrift(ENVDRIFT_RATE=1, genome_shape=(3, 4))
        assert ed.map.sum() == 0
        ed.evolve(step=1)
        assert ed.map.sum() == 1

    def test_evolve_skips_off_rate(self):
        """Evolving at a non-rate step leaves the map unchanged."""
        ed = Envdrift(ENVDRIFT_RATE=10, genome_shape=(3, 4))
        ed.evolve(step=7)
        assert ed.map.sum() == 0


class TestCall:
    """Verify call() XORs the map with the input array."""

    def test_no_drift_passthrough(self):
        """With rate=0 (no map), call returns the input unchanged."""
        ed = Envdrift(ENVDRIFT_RATE=0, genome_shape=(2, 3))
        arr = np.ones((5, 2, 3), dtype=bool)
        result = ed.call(arr)
        np.testing.assert_array_equal(result, arr)

    def test_xor_flips_bits(self):
        """Map with a True bit flips that position in the input."""
        ed = Envdrift(ENVDRIFT_RATE=10, genome_shape=(3,))
        ed.map = np.array([True, False, False], dtype=bool)
        arr = np.array([[True, True, True]], dtype=bool)
        result = ed.call(arr)
        expected = np.array([[False, True, True]], dtype=bool)
        np.testing.assert_array_equal(result, expected)

    def test_xor_double_flip_restores(self):
        """XOR with the same map twice restores the original."""
        ed = Envdrift(ENVDRIFT_RATE=10, genome_shape=(4,))
        ed.map = np.array([True, True, False, False], dtype=bool)
        arr = np.array([[True, False, True, False]], dtype=bool)
        result = ed.call(ed.call(arr))
        np.testing.assert_array_equal(result, arr)
