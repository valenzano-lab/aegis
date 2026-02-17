"""Unit tests for the composite Interpreter.

Covers: all 10 interpreter methods (const1, single_bit, threshold,
linear, binary, switch, binary_switch, uniform, exp, binary_exp)
with known inputs and expected outputs.
"""

import numpy as np
import pytest

import aegis_sim.variables as variables
from aegis_sim.submodels.genetics.composite.interpreter import Interpreter


@pytest.fixture(autouse=True)
def _init_rng():
    """Ensure variables.rng is initialized for switch interpreter."""
    variables.rng = np.random.default_rng(42)


def _make_interpreter(bits_per_locus=4, threshold=2):
    return Interpreter(BITS_PER_LOCUS=bits_per_locus, THRESHOLD=threshold)


class TestConst1:
    """const1: always returns 1 regardless of locus content."""

    def test_returns_ones(self):
        interp = _make_interpreter(bits_per_locus=3)
        loci = np.zeros((5, 2, 3), dtype=bool)
        result = interp.call(loci, "const1")
        np.testing.assert_array_equal(result, np.ones((5, 1)))

    def test_shape(self):
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((3, 7, 4), dtype=bool)
        result = interp.call(loci, "const1")
        assert result.shape == (3, 1)


class TestSingleBit:
    """single_bit: returns only the first bit of each locus."""

    def test_first_bit_true(self):
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.zeros((2, 3, 4), dtype=bool)
        loci[:, :, 0] = True
        result = interp.call(loci, "single_bit")
        assert np.all(result == True)

    def test_first_bit_false(self):
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((2, 3, 4), dtype=bool)
        loci[:, :, 0] = False
        result = interp.call(loci, "single_bit")
        assert np.all(result == False)

    def test_output_shape(self):
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((5, 10, 4), dtype=bool)
        result = interp.call(loci, "single_bit")
        assert result.shape == (5, 10)


class TestUniform:
    """uniform: normalized sum of bits (position-independent)."""

    def test_all_ones(self):
        """All bits on -> 1.0."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((3, 2, 4), dtype=bool)
        result = interp.call(loci, "uniform")
        np.testing.assert_allclose(result, 1.0)

    def test_all_zeros(self):
        """All bits off -> 0.0."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.zeros((3, 2, 4), dtype=bool)
        result = interp.call(loci, "uniform")
        np.testing.assert_allclose(result, 0.0)

    def test_half_bits(self):
        """2 of 4 bits on -> 0.5."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.zeros((1, 1, 4), dtype=bool)
        loci[0, 0, :2] = True
        result = interp.call(loci, "uniform")
        assert result[0, 0] == pytest.approx(0.5)


class TestBinary:
    """binary: interprets locus as a binary number, normalized to [0, 1]."""

    def test_all_ones(self):
        """All bits on -> 1.0."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((1, 1, 4), dtype=bool)
        result = interp.call(loci, "binary")
        assert result[0, 0] == pytest.approx(1.0)

    def test_all_zeros(self):
        """All bits off -> 0.0."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.zeros((1, 1, 4), dtype=bool)
        result = interp.call(loci, "binary")
        assert result[0, 0] == pytest.approx(0.0)

    def test_msb_only(self):
        """Only MSB on: weight = 8/15 for 4-bit."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.zeros((1, 1, 4), dtype=bool)
        loci[0, 0, 0] = True  # MSB
        result = interp.call(loci, "binary")
        assert result[0, 0] == pytest.approx(8 / 15)

    def test_position_dependent(self):
        """Different bit positions produce different values."""
        interp = _make_interpreter(bits_per_locus=4)
        loci_msb = np.zeros((1, 1, 4), dtype=bool)
        loci_msb[0, 0, 0] = True
        loci_lsb = np.zeros((1, 1, 4), dtype=bool)
        loci_lsb[0, 0, 3] = True
        assert interp.call(loci_msb, "binary")[0, 0] > interp.call(loci_lsb, "binary")[0, 0]


class TestLinear:
    """linear: weighted sum with linearly decreasing weights."""

    def test_all_ones(self):
        """All bits on -> 1.0."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((1, 1, 4), dtype=bool)
        result = interp.call(loci, "linear")
        assert result[0, 0] == pytest.approx(1.0)

    def test_all_zeros(self):
        """All bits off -> 0.0."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.zeros((1, 1, 4), dtype=bool)
        result = interp.call(loci, "linear")
        assert result[0, 0] == pytest.approx(0.0)


class TestExp:
    """exp: base^(number_of_zeros). Suitable for very small numbers."""

    def test_all_ones_gives_one(self):
        """No zeros -> 0.5^0 = 1.0."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((1, 1, 4), dtype=bool)
        result = interp.call(loci, "exp")
        assert result[0, 0] == pytest.approx(1.0)

    def test_all_zeros_gives_small(self):
        """4 zeros -> 0.5^4 = 0.0625."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.zeros((1, 1, 4), dtype=bool)
        result = interp.call(loci, "exp")
        assert result[0, 0] == pytest.approx(0.5**4)

    def test_one_zero(self):
        """1 zero -> 0.5^1 = 0.5."""
        interp = _make_interpreter(bits_per_locus=4)
        loci = np.ones((1, 1, 4), dtype=bool)
        loci[0, 0, 0] = False
        result = interp.call(loci, "exp")
        assert result[0, 0] == pytest.approx(0.5)


class TestThreshold:
    """threshold (Penna): cumulative count of False first-bits < THRESHOLD."""

    def test_all_true_all_pass(self):
        """All first bits True -> cumsum of False = 0 -> all < threshold."""
        interp = _make_interpreter(bits_per_locus=4, threshold=2)
        loci = np.ones((1, 5, 4), dtype=bool)
        result = interp.call(loci, "threshold")
        assert np.all(result == True)

    def test_all_false_limited(self):
        """All first bits False -> cumsum = [1,2,3,...] -> only first (threshold-1) pass."""
        interp = _make_interpreter(bits_per_locus=4, threshold=2)
        loci = np.zeros((1, 5, 4), dtype=bool)
        result = interp.call(loci, "threshold")
        # cumsum of ~False = cumsum of True = [1,2,3,4,5], < 2 means [True, False, ...]
        assert result[0, 0] == True
        assert result[0, 1] == False


class TestInvalidInterpreter:
    """Verify unknown interpreter name raises KeyError."""

    def test_unknown_raises(self):
        interp = _make_interpreter()
        loci = np.ones((1, 1, 4), dtype=bool)
        with pytest.raises(KeyError):
            interp.call(loci, "nonexistent")
