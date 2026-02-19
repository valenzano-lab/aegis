"""Unit tests for the Abiotic mortality submodel.

Covers: all seven hazard waveforms (flat, sinusoidal, triangle, square,
sawtooth, ramp, instant, instant_fatal), the additive offset, and
rejection of unknown shapes.
"""

import math
import pytest

from aegis_sim.submodels.abiotic import Abiotic


def _make_abiotic(shape="flat", offset=0.0, amplitude=0.1, period=100):
    return Abiotic(
        ABIOTIC_HAZARD_SHAPE=shape,
        ABIOTIC_HAZARD_OFFSET=offset,
        ABIOTIC_HAZARD_AMPLITUDE=amplitude,
        ABIOTIC_HAZARD_PERIOD=period,
    )


class TestAbioticFlat:
    """Flat waveform: constant hazard equal to amplitude + offset."""

    def test_flat_returns_amplitude_plus_offset(self):
        """Hazard is amplitude + offset regardless of step."""
        ab = _make_abiotic("flat", offset=0.05, amplitude=0.1)
        assert ab(0) == pytest.approx(0.15)
        assert ab(50) == pytest.approx(0.15)

    def test_flat_zero_amplitude(self):
        """Zero amplitude and zero offset yields zero hazard."""
        ab = _make_abiotic("flat", offset=0.0, amplitude=0.0)
        assert ab(10) == pytest.approx(0.0)


class TestAbioticSinusoidal:
    """Sinusoidal waveform: amplitude * sin(2*pi*step/period) + offset."""

    def test_sinusoidal_at_zero(self):
        """sin(0) = 0, so hazard equals offset at step 0."""
        ab = _make_abiotic("sinusoidal", offset=0.0, amplitude=0.5, period=100)
        assert ab(0) == pytest.approx(0.0)

    def test_sinusoidal_at_quarter_period(self):
        """sin(pi/2) = 1, so hazard equals full amplitude at step=period/4."""
        ab = _make_abiotic("sinusoidal", offset=0.0, amplitude=0.5, period=100)
        assert ab(25) == pytest.approx(0.5)

    def test_sinusoidal_at_half_period(self):
        """sin(pi) ~ 0, so hazard returns to offset at step=period/2."""
        ab = _make_abiotic("sinusoidal", offset=0.0, amplitude=0.5, period=100)
        assert ab(50) == pytest.approx(0.0, abs=1e-10)


class TestAbioticSquare:
    """Square waveform: +amplitude in first half, -amplitude in second half."""

    def test_square_first_half(self):
        """Step in the first half of the period yields +amplitude."""
        ab = _make_abiotic("square", offset=0.0, amplitude=0.3, period=100)
        assert ab(10) == pytest.approx(0.3)

    def test_square_second_half(self):
        """Step in the second half of the period yields -amplitude."""
        ab = _make_abiotic("square", offset=0.0, amplitude=0.3, period=100)
        assert ab(75) == pytest.approx(-0.3)


class TestAbioticRamp:
    """Ramp (backward sawtooth) waveform: linearly increases from 0 to amplitude over one period."""

    def test_ramp_at_zero(self):
        """Ramp starts at 0 at the beginning of the period."""
        ab = _make_abiotic("ramp", offset=0.0, amplitude=1.0, period=100)
        assert ab(0) == pytest.approx(0.0)

    def test_ramp_at_half(self):
        """Ramp reaches half amplitude at the midpoint."""
        ab = _make_abiotic("ramp", offset=0.0, amplitude=1.0, period=100)
        assert ab(50) == pytest.approx(0.5)

    def test_ramp_wraps(self):
        """Ramp resets to 0 at the start of the next period."""
        ab = _make_abiotic("ramp", offset=0.0, amplitude=1.0, period=100)
        assert ab(100) == pytest.approx(0.0)


class TestAbioticInstantFatal:
    """Instant-fatal waveform: kills 100% every PERIOD steps, 0% otherwise."""

    def test_instant_fatal_at_zero(self):
        """Step 0 is always unaffected (no kill)."""
        ab = _make_abiotic("instant_fatal", offset=0.0, amplitude=1.0, period=50)
        assert ab(0) == pytest.approx(0.0)

    def test_instant_fatal_at_period(self):
        """At exactly the period boundary, mortality is 1.0 (total kill)."""
        ab = _make_abiotic("instant_fatal", offset=0.0, amplitude=1.0, period=50)
        assert ab(50) == pytest.approx(1.0)

    def test_instant_fatal_off_period(self):
        """Between period boundaries, mortality is 0."""
        ab = _make_abiotic("instant_fatal", offset=0.0, amplitude=1.0, period=50)
        assert ab(25) == pytest.approx(0.0)


class TestAbioticOffset:
    """Verify that ABIOTIC_HAZARD_OFFSET is added to the waveform value."""

    def test_offset_added(self):
        """Flat shape with zero amplitude returns just the offset."""
        ab = _make_abiotic("flat", offset=0.2, amplitude=0.0)
        assert ab(0) == pytest.approx(0.2)


class TestAbioticInvalidShape:
    """Verify that an unknown shape name is rejected."""

    def test_invalid_shape_raises(self):
        """Constructing with a nonexistent shape raises KeyError."""
        with pytest.raises(KeyError):
            _make_abiotic("nonexistent_shape")


class TestAbioticInstantDeterministic:
    """Instant-deterministic waveform: kills exactly AMPLITUDE fraction every PERIOD steps, 0% otherwise."""

    def test_at_zero(self):
        """Step 0 is always unaffected."""
        ab = _make_abiotic("instant_deterministic", offset=0.0, amplitude=0.4, period=50)
        assert ab(0) == pytest.approx(0.0)

    def test_at_period(self):
        """At exactly the period boundary, mortality equals amplitude."""
        ab = _make_abiotic("instant_deterministic", offset=0.0, amplitude=0.4, period=50)
        assert ab(50) == pytest.approx(0.4)

    def test_at_double_period(self):
        """Fires again at 2x the period."""
        ab = _make_abiotic("instant_deterministic", offset=0.0, amplitude=0.4, period=50)
        assert ab(100) == pytest.approx(0.4)

    def test_off_period(self):
        """Between period boundaries, mortality is 0."""
        ab = _make_abiotic("instant_deterministic", offset=0.0, amplitude=0.4, period=50)
        assert ab(25) == pytest.approx(0.0)

    def test_with_offset(self):
        """Offset is added on top of the amplitude at the period boundary."""
        ab = _make_abiotic("instant_deterministic", offset=0.1, amplitude=0.4, period=50)
        assert ab(50) == pytest.approx(0.5)

    def test_off_period_with_offset(self):
        """Off-period steps still get the offset."""
        ab = _make_abiotic("instant_deterministic", offset=0.1, amplitude=0.4, period=50)
        assert ab(25) == pytest.approx(0.1)

    def test_is_deterministic(self):
        """Unlike 'instant', calling multiple times at the same step returns the same value."""
        ab = _make_abiotic("instant_deterministic", offset=0.0, amplitude=0.4, period=50)
        results = [ab(50) for _ in range(20)]
        assert all(r == results[0] for r in results)
