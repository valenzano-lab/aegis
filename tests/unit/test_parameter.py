"""Unit tests for the Parameter class.

Covers: convert() type coercion (None, empty string, int, float, bool),
valid() type and range checking, validate_dtype() including int-for-float
and None handling, validate_inrange(), and get_name() with explicit vs
inferred names.
"""

import pytest

from aegis_sim.parameterization.parameter import Parameter


def _make_param(**overrides):
    defaults = dict(
        key="TEST_PARAM",
        name="Test Parameter",
        domain="test",
        default=10,
        info="A test parameter",
        dtype=int,
        drange="0-100",
        inrange=lambda x: 0 <= x <= 100,
    )
    defaults.update(overrides)
    return Parameter(**defaults)


class TestParameterConvert:
    """Verify convert() coerces raw input to the parameter's dtype."""

    def test_convert_none_returns_default(self):
        """None input falls back to the default value."""
        p = _make_param(default=42)
        assert p.convert(None) == 42

    def test_convert_empty_string_returns_default(self):
        """Empty string input falls back to the default value."""
        p = _make_param(default=42)
        assert p.convert("") == 42

    def test_convert_int(self):
        """String '7' is converted to int 7."""
        p = _make_param(dtype=int)
        assert p.convert("7") == 7

    def test_convert_float(self):
        """String '3.14' is converted to float 3.14."""
        p = _make_param(dtype=float, default=1.0)
        assert p.convert("3.14") == pytest.approx(3.14)

    def test_convert_bool_true(self):
        """'True', 'true', and True all convert to True."""
        p = _make_param(dtype=bool, default=False)
        assert p.convert("True") is True
        assert p.convert("true") is True
        assert p.convert(True) is True

    def test_convert_bool_false(self):
        """'False' and arbitrary strings convert to False."""
        p = _make_param(dtype=bool, default=False)
        assert p.convert("False") is False
        assert p.convert("anything") is False


class TestParameterValid:
    """Verify valid() checks both type and range."""

    def test_valid_value(self):
        """In-range int passes validation."""
        p = _make_param()
        assert p.valid(50) is True

    def test_wrong_type(self):
        """String value for an int parameter fails validation."""
        p = _make_param(dtype=int)
        assert p.valid("not_an_int") is False

    def test_out_of_range(self):
        """Value outside inrange predicate fails; inside passes."""
        p = _make_param(inrange=lambda x: x > 0)
        assert p.valid(0) is False
        assert p.valid(1) is True


class TestParameterValidateDtype:
    """Verify validate_dtype() raises TypeError on mismatches."""

    def test_correct_type_passes(self):
        """Matching dtype does not raise."""
        p = _make_param(dtype=int)
        p.validate_dtype(5)

    def test_wrong_type_raises(self):
        """Mismatched dtype raises TypeError."""
        p = _make_param(dtype=int)
        with pytest.raises(TypeError):
            p.validate_dtype("string")

    def test_int_accepted_for_float(self):
        """int is accepted when the parameter expects float."""
        p = _make_param(dtype=float, default=1.0)
        p.validate_dtype(5)

    def test_none_accepted_when_default_is_none(self):
        """None is valid when the parameter's default is None."""
        p = _make_param(default=None, dtype=int)
        p.validate_dtype(None)

    def test_none_rejected_when_default_not_none(self):
        """None raises TypeError when the default is not None."""
        p = _make_param(default=10, dtype=int)
        with pytest.raises(TypeError):
            p.validate_dtype(None)


class TestParameterValidateInrange:
    """Verify validate_inrange() raises ValueError when out of range."""

    def test_in_range_passes(self):
        """Value satisfying inrange does not raise."""
        p = _make_param(inrange=lambda x: x > 0)
        p.validate_inrange(5)

    def test_out_of_range_raises(self):
        """Value violating inrange raises ValueError."""
        p = _make_param(inrange=lambda x: x > 0)
        with pytest.raises(ValueError):
            p.validate_inrange(-1)


class TestParameterGetName:
    """Verify get_name() returns the explicit name or infers one from the key."""

    def test_explicit_name(self):
        """When name is set, get_name() returns it directly."""
        p = _make_param(name="My Param")
        assert p.get_name() == "My Param"

    def test_inferred_from_key(self):
        """When name is None, get_name() lowercases and de-underscores the key."""
        p = _make_param(key="SOME_COOL_PARAM", name=None)
        assert p.get_name() == "some cool param"
