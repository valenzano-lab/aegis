"""
Container tests — runs its own simulation, then tests Container against the output.
"""

import pytest
import yaml
import pandas as pd

from aegis_sim import run
from aegis_sim.utilities.container import Container


@pytest.fixture(scope="module")
def sim_output(tmp_path_factory):
    """Run a single short simulation and return the output directory."""
    tmp = tmp_path_factory.mktemp("container_test")
    config = {
        "STEPS_PER_SIMULATION": 100,
        "LOGGING_RATE": 10,
        "INTERVAL_RATE": 25,
        "POPGENSTATS_RATE": 50,
        "TE_RATE": 50,
        "TE_DURATION": 25,
        "SNAPSHOT_RATE": 50,
        "PICKLE_RATE": 50,
        "SNAPSHOT_FINAL_COUNT": 3,
        "INITIAL_POPULATION_SIZE": 200,
    }
    config_path = tmp / "container_sim.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )
    return tmp / "container_sim"


@pytest.fixture
def container(sim_output):
    return Container(sim_output)


# --- Path resolution ---

def test_get_path_core_files(container):
    container.get_path(name="genotypes")
    container.get_path(name="phenotypes")
    container.get_path(name="log")
    container.get_path(name="ticker")
    container.get_path(name="output_summary")
    container.get_path(name="input_summary")
    container.get_path(name="popsize_before_reproduction")
    container.get_path(name="popsize_after_reproduction")
    container.get_path(name="eggnum_after_reproduction")


def test_get_path_optional_files(container):
    container.get_path(name="envdriftmap")
    container.get_path(name="snapshots")
    container.get_path(name="pickles")


# --- Record structure ---

def test_get_record_structure(container):
    _ = container.get_record_structure()


# --- Metadata ---

def test_get_log(container):
    result = container.get_log()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "step" in result.columns or result.shape[1] > 0


def test_get_simple_log(container):
    result = container.get_simple_log()
    assert result is not None


def test_get_ticker(container):
    ticker = container.get_ticker()
    assert ticker is not None
    timestamp = ticker.read()
    assert timestamp is not None
    assert len(timestamp) == 19  # YYYY-MM-DD HH:MM:SS


def test_get_config(container):
    result = container.get_config()
    assert isinstance(result, dict)
    assert "STEPS_PER_SIMULATION" in result
    assert result["STEPS_PER_SIMULATION"] == 100


def test_get_final_config(container):
    result = container.get_final_config()
    assert isinstance(result, dict)
    assert "STEPS_PER_SIMULATION" in result


def test_get_output_summary(container):
    result = container.get_output_summary()
    assert result is not None


def test_get_input_summary(container):
    result = container.get_input_summary()
    assert result is not None


# --- Demography ---

def test_get_generations_until_interval(container):
    _ = container.get_generations_until_interval()


def test_get_birth_table_observed_interval(container):
    _ = container.get_birth_table_observed_interval()


def test_get_life_table_observed_interval(container):
    _ = container.get_life_table_observed_interval()


def test_get_life_table_observed_snapshot(container):
    _ = container.get_life_table_observed_snapshot(record_index=0)
    _ = container.get_life_table_observed_snapshot(record_index=-1)


def test_get_death_table_observed_interval(container):
    _ = container.get_death_table_observed_interval()


def test_get_surv_observed_interval(container):
    _ = container.get_surv_observed_interval()


def test_get_fert_observed_interval(container):
    _ = container.get_fert_observed_interval()


# --- Population size ---

def test_get_population_size_before_reproduction(container):
    result = container.get_population_size_before_reproduction()
    assert isinstance(result, pd.DataFrame) or hasattr(result, '__len__')
    assert len(result) == 100  # one entry per step


def test_get_population_size_after_reproduction(container):
    result = container.get_population_size_after_reproduction()
    assert len(result) == 100


def test_get_egg_number_after_reproduction(container):
    result = container.get_egg_number_after_reproduction()
    assert len(result) == 100


# --- Resources ---

def test_get_resource_amount_before_scavenging(container):
    result = container.get_resource_amount_before_scavenging()
    assert len(result) == 100


def test_get_resource_amount_after_scavenging(container):
    result = container.get_resource_amount_after_scavenging()
    assert len(result) == 100


# --- Genomics / Phenomics ---

def test_get_genotypes_intrinsic_snapshot(container):
    result = container.get_genotypes_intrinsic_snapshot(record_index=0)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0, "Snapshot should contain at least one individual"

    result_last = container.get_genotypes_intrinsic_snapshot(record_index=-1)
    assert isinstance(result_last, pd.DataFrame)
    assert len(result_last) > 0


def test_get_phenotype_intrinsic_snapshot(container):
    result = container.get_phenotype_intrinsic_snapshot(record_index=0, trait=None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

    result_last = container.get_phenotype_intrinsic_snapshot(record_index=-1, trait=None)
    assert len(result_last) > 0


def test_get_demography_observed_snapshot(container):
    result = container.get_demography_observed_snapshot(record_index=0)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

    result_last = container.get_demography_observed_snapshot(record_index=-1)
    assert len(result_last) > 0


def test_get_genotypes_intrinsic_interval(container):
    result = container.get_genotypes_intrinsic_interval()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0, "Interval genotypes should have data rows"


def test_get_phenotype_intrinsic_interval_surv(container):
    result = container.get_phenotype_intrinsic_interval(trait="surv")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    # Phenotype values should be in [0, 1] range
    numeric_cols = result.select_dtypes(include="number")
    if len(numeric_cols.columns) > 0:
        assert numeric_cols.min().min() >= 0, "Phenotype values should be >= 0"
        assert numeric_cols.max().max() <= 1, "Phenotype values should be <= 1"


def test_get_phenotype_intrinsic_interval_repr(container):
    result = container.get_phenotype_intrinsic_interval(trait="repr")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_get_phenotype_intrinsic_interval_optional_traits(container):
    """muta and grow are only present if their G_*_evolvable param is True."""
    df = pd.read_csv(container.get_path("phenotypes"), header=[0, 1], nrows=0)
    available_traits = df.columns.get_level_values(0).unique().tolist()
    if "muta" in available_traits:
        _ = container.get_phenotype_intrinsic_interval(trait="muta")
    if "grow" in available_traits:
        _ = container.get_phenotype_intrinsic_interval(trait="grow")


def test_get_envidriftmap(container):
    _ = container.get_envidriftmap()


# --- Reproduction stats ---

def test_get_lifetime_reproduction(container):
    _ = container.get_lifetime_reproduction()


def test_get_average_age_at_reproduction(container):
    _ = container.get_average_age_at_reproduction()
