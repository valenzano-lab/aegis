import pathlib

import numpy as np
import pytest

import aegis_sim
from aegis_sim import parameterization
from aegis_sim.dataclasses.phenotypes import Phenotypes
from aegis_sim.submodels.genetics import selectable_surv

BROADCAST_CONFIG = pathlib.Path(__file__).absolute().parent / "test_selectable_via_surv_broadcast.yml"
MATCHED_CONFIG = pathlib.Path(__file__).absolute().parent / "test_selectable_via_surv_matched.yml"
MISMATCH_CONFIG = pathlib.Path(__file__).absolute().parent / "test_selectable_via_surv_mismatch.yml"


def _make_phenotypes(surv_values, selectable_values):
    """Build a Phenotypes object with explicit surv / selectable_via_surv columns.

    G_surv_lo/hi and G_selectable_via_surv_lo/hi are 0/1 in the test configs, so
    Phenotypes.__init__'s lo/hi rescale is the identity and these values pass through
    unchanged.
    """
    n = surv_values.shape[0]
    array = np.zeros((n, parameterization.expected_phenotype_length), dtype=np.float32)
    surv = parameterization.traits["surv"]
    trait = parameterization.traits["selectable_via_surv"]
    array[:, surv.slice] = surv_values
    array[:, trait.slice] = selectable_values
    return Phenotypes(array)


def test_selectable_via_surv_is_a_first_class_visible_trait():
    """It must have its own genome loci and its own phenotype, extractable like any
    other trait -- not just an internal implementation detail."""
    aegis_sim.init(custom_config_path=BROADCAST_CONFIG, overwrite=True)

    trait = parameterization.traits["selectable_via_surv"]
    assert trait.evolvable
    assert trait.length == 1  # agespecific=False

    from aegis_sim import submodels
    from aegis_sim.dataclasses.bitarray import Genomes

    architecture = submodels.architect.architecture
    genomes = Genomes(architecture.init_genome_array(popsize=20))
    phenotypes = submodels.architect(genomes)

    ages = np.zeros(20, dtype=np.int32)
    values = phenotypes.extract("selectable_via_surv", ages)
    assert values.shape == (20,)
    # Values must vary across the population (it's a real, heritable, evolved trait)
    assert len(set(np.round(values, 3))) > 1


def test_directional_selection_broadcasts_single_value_to_all_surv_ages():
    aegis_sim.init(custom_config_path=BROADCAST_CONFIG, overwrite=True)

    surv_values = np.full((4, 3), 0.5, dtype=np.float32)  # AGE_LIMIT=3
    selectable_values = np.array([[0.7], [0.3], [0.5], [1.0]], dtype=np.float32)

    phenotypes = _make_phenotypes(surv_values, selectable_values)
    phenotypes = selectable_surv.apply(phenotypes)

    surv = parameterization.traits["surv"]
    result = phenotypes.array[:, surv.slice]

    # cutoff=0.5, benefit=+0.1, penalty=-0.2; >= cutoff counts as benefit
    expected = np.array(
        [
            [0.6, 0.6, 0.6],  # 0.7 >= 0.5 -> benefit
            [0.3, 0.3, 0.3],  # 0.3 < 0.5 -> penalty (0.5 - 0.2)
            [0.6, 0.6, 0.6],  # 0.5 >= 0.5 (tie counts as benefit)
            [0.6, 0.6, 0.6],  # 1.0 >= 0.5 -> benefit
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)


def test_surv_hard_clipped_to_0_1():
    aegis_sim.init(custom_config_path=BROADCAST_CONFIG, overwrite=True)

    surv_values = np.array([[0.95, 0.95, 0.95], [0.05, 0.05, 0.05]], dtype=np.float32)
    selectable_values = np.array([[0.9], [0.1]], dtype=np.float32)  # benefit, penalty

    phenotypes = _make_phenotypes(surv_values, selectable_values)
    phenotypes = selectable_surv.apply(phenotypes)

    surv = parameterization.traits["surv"]
    result = phenotypes.array[:, surv.slice]

    # 0.95 + 0.1 = 1.05 -> clipped to 1.0; 0.05 - 0.2 = -0.15 -> clipped to 0.0
    assert np.allclose(result[0], 1.0)
    assert np.allclose(result[1], 0.0)


def test_stabilizing_selection_gaussian_within_sd_and_constant_beyond():
    aegis_sim.init(custom_config_path=MATCHED_CONFIG, overwrite=True)

    # mean=0.5, sd=0.1, max_benefit=0.2, const_penalty=-0.3, beyond_sd=2
    surv_values = np.zeros((3, 3), dtype=np.float32)
    selectable_values = np.array(
        [
            [0.5, 0.5, 0.5],  # at the mean -> full max_benefit
            [0.6, 0.6, 0.6],  # 1 sd away -> partial gaussian benefit
            [0.9, 0.9, 0.9],  # 4 sd away -> beyond threshold -> constant penalty
        ],
        dtype=np.float32,
    )

    phenotypes = _make_phenotypes(surv_values, selectable_values)
    phenotypes = selectable_surv.apply(phenotypes)

    surv = parameterization.traits["surv"]
    result = phenotypes.array[:, surv.slice]

    expected_at_mean = 0.2  # max_benefit * exp(0) = max_benefit
    expected_at_1sd = 0.2 * np.exp(-0.5 * 1**2)
    expected_beyond = -0.3  # clipped to 0 since surv baseline is 0 and penalty is negative

    assert np.allclose(result[0], expected_at_mean)
    assert np.allclose(result[1], expected_at_1sd)
    assert np.allclose(result[2], 0.0)  # 0 + (-0.3) clipped to 0


def test_matched_agespecific_lengths_apply_delta_per_age():
    aegis_sim.init(custom_config_path=MATCHED_CONFIG, overwrite=True)

    trait = parameterization.traits["selectable_via_surv"]
    surv = parameterization.traits["surv"]
    assert trait.length == surv.length == 3

    surv_values = np.zeros((1, 3), dtype=np.float32)
    selectable_values = np.array([[0.5, 0.9, 0.5]], dtype=np.float32)  # mean, far, mean

    phenotypes = _make_phenotypes(surv_values, selectable_values)
    phenotypes = selectable_surv.apply(phenotypes)

    result = phenotypes.array[:, surv.slice][0]
    assert np.allclose(result[0], 0.2)  # at mean -> max_benefit
    assert np.allclose(result[1], 0.0)  # far -> const_penalty (-0.3), clipped to 0
    assert np.allclose(result[2], 0.2)


def test_validate_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="selectable_via_surv has"):
        aegis_sim.init(custom_config_path=MISMATCH_CONFIG, overwrite=True)


def test_inactive_by_default_is_a_no_op():
    """When G_selectable_via_surv_evolvable is left at its default (False), the
    trait must not exist as a genome-encoded trait and apply() must be a no-op."""
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump({"STEPS_PER_SIMULATION": 1, "LOGGING_RATE": 1, "INITIAL_POPULATION_SIZE": 5}, f)
        path = pathlib.Path(f.name)

    aegis_sim.init(custom_config_path=path, overwrite=True)
    assert not parameterization.traits["selectable_via_surv"].evolvable
    assert not selectable_surv.is_active()

    trait = parameterization.traits["selectable_via_surv"]
    assert trait.length == 0  # not evolvable -> no genome loci at all

    surv_values = np.full((2, parameterization.parametermanager.parameters.AGE_LIMIT), 0.5, dtype=np.float32)
    phenotypes = _make_phenotypes(surv_values, np.zeros((2, trait.length), dtype=np.float32))
    before = phenotypes.array.copy()
    phenotypes = selectable_surv.apply(phenotypes)
    assert np.array_equal(phenotypes.array, before)
