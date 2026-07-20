"""Parameter overrides on resume.

Resume restores parameters from the checkpoint, which makes a two-phase experiment
inexpressible: you cannot burn a population in to equilibrium under one regime and
then run it under another. `--override KEY=VALUE` fills that gap, so several arms can
resume the SAME equilibrated checkpoint under different regimes and differ only in the
regime -- not in their burn-in history.

Structural parameters must be rejected: they define the shape of the checkpointed
genome/phenotype arrays, so changing them would leave the restored arrays inconsistent
with the parameters describing them.
"""

import pytest

from aegis import parse_overrides
from aegis_sim.parameterization.parametermanager import ParameterManager


class TestParseOverrides:

    def test_empty(self):
        assert parse_overrides(None) == {}
        assert parse_overrides([]) == {}

    def test_casts_to_declared_dtype(self):
        out = parse_overrides(["RESOURCE_ADDITIVE_GROWTH=250"])
        assert out == {"RESOURCE_ADDITIVE_GROWTH": 250.0}
        assert isinstance(out["RESOURCE_ADDITIVE_GROWTH"], float)

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("True", True), ("1", True),
        ("false", False), ("False", False), ("0", False),
    ])
    def test_boolean_forms(self, raw, expected):
        assert parse_overrides([f"RESOURCE_DEFICIT_CARRYOVER={raw}"]) == {
            "RESOURCE_DEFICIT_CARRYOVER": expected}

    def test_rejects_non_boolean_for_bool_param(self):
        with pytest.raises(ValueError, match="expects a boolean"):
            parse_overrides(["RESOURCE_DEFICIT_CARRYOVER=maybe"])

    def test_inf_for_uncapped_pool(self):
        out = parse_overrides(["RESOURCE_MAXIMUM_AMOUNT=inf"])
        assert out["RESOURCE_MAXIMUM_AMOUNT"] == float("inf")

    def test_none_allowed_where_the_parameter_defaults_to_none(self):
        assert parse_overrides(["CARRYING_CAPACITY_EGGS=None"]) == {"CARRYING_CAPACITY_EGGS": None}

    def test_rejects_unknown_parameter(self):
        with pytest.raises(ValueError, match="not a valid parameter name"):
            parse_overrides(["NOT_A_PARAM=1"])

    def test_rejects_missing_equals(self):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            parse_overrides(["RESOURCE_ADDITIVE_GROWTH"])

    def test_multiple_overrides_accumulate(self):
        out = parse_overrides([
            "RESOURCE_ADDITIVE_GROWTH=100",
            "RESOURCE_DEFICIT_CARRYOVER=true",
            "RESOURCE_MAXIMUM_AMOUNT=inf",
        ])
        assert len(out) == 3


class TestStructuralGuard:

    def _config(self):
        return {"AGE_LIMIT": 50, "BITS_PER_LOCUS": 20, "RESOURCE_ADDITIVE_GROWTH": 500.0,
                "RESOURCE_DEFICIT_CARRYOVER": False, "RANDOM_SEED": 1}

    def test_non_structural_override_is_applied(self):
        pm = ParameterManager()
        pm.init_from_config(self._config(), "", overrides={"RESOURCE_ADDITIVE_GROWTH": 250.0})
        assert pm.parameters.RESOURCE_ADDITIVE_GROWTH == 250.0
        assert pm.final_config["RESOURCE_ADDITIVE_GROWTH"] == 250.0, "must be recorded for reproducibility"

    def test_untouched_parameters_survive(self):
        pm = ParameterManager()
        pm.init_from_config(self._config(), "", overrides={"RESOURCE_ADDITIVE_GROWTH": 250.0})
        assert pm.parameters.AGE_LIMIT == 50

    def test_no_overrides_is_unchanged(self):
        pm = ParameterManager()
        pm.init_from_config(self._config(), "")
        assert pm.parameters.RESOURCE_ADDITIVE_GROWTH == 500.0

    @pytest.mark.parametrize("key,value", [
        ("AGE_LIMIT", 60),
        ("BITS_PER_LOCUS", 10),
        ("PLOIDY", 1),
        ("GENARCH_TYPE", "modifying"),
        ("REPRODUCTION_MODE", "asexual"),
        ("G_surv_evolvable", False),
        ("G_repr_agespecific", False),
    ])
    def test_structural_overrides_are_rejected(self, key, value):
        """These change array shapes; the checkpointed genomes would no longer match."""
        pm = ParameterManager()
        with pytest.raises(ValueError, match="Cannot override"):
            pm.init_from_config(self._config(), "", overrides={key: value})

    def test_random_seed_rejected_because_rng_state_is_restored(self):
        """Not structural, but a new seed would be silently ignored — misleading."""
        pm = ParameterManager()
        with pytest.raises(ValueError, match="Cannot override"):
            pm.init_from_config(self._config(), "", overrides={"RANDOM_SEED": 99})

    def test_rejection_lists_every_offending_key(self):
        pm = ParameterManager()
        with pytest.raises(ValueError) as e:
            pm.init_from_config(self._config(), "",
                                overrides={"AGE_LIMIT": 60, "PLOIDY": 1,
                                           "RESOURCE_ADDITIVE_GROWTH": 100.0})
        assert "AGE_LIMIT" in str(e.value) and "PLOIDY" in str(e.value)

    def test_invalid_value_rejected_by_parameter_validation(self):
        pm = ParameterManager()
        with pytest.raises(TypeError):
            pm.init_from_config(self._config(), "", overrides={"RESOURCE_ADDITIVE_GROWTH": "lots"})
