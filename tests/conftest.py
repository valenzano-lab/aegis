"""Shared fixtures for functional tests."""

import pytest
import yaml
import pathlib

# Shared experiment path used by legacy functional tests
test_experiment_path = pathlib.Path(__file__).absolute().parent / "functional" / "experiments"
test_experiment_path.mkdir(exist_ok=True, parents=True)


@pytest.fixture
def base_config():
    """Minimal config dict for a short simulation."""
    return {
        "STEPS_PER_SIMULATION": 100,
        "LOGGING_RATE": 10,
        "INTERVAL_RATE": 25,
        "POPGENSTATS_RATE": 50,
        "TE_RATE": 50,
        "TE_DURATION": 25,
        "SNAPSHOT_RATE": 50,
        "PICKLE_RATE": 50,
        "INITIAL_POPULATION_SIZE": 200,
        "CARRYING_CAPACITY_EGGS": 500,
    }


@pytest.fixture
def checkpoint_config(base_config):
    """Config with checkpointing enabled."""
    return {**base_config, "CHECKPOINT_RATE": 25}


@pytest.fixture
def write_config(tmp_path):
    """Factory fixture: write a config dict to a .yml file and return its path."""
    def _write(config: dict, name: str = "test_sim") -> pathlib.Path:
        config_path = tmp_path / f"{name}.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path
    return _write
