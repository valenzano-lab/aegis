"""Unit tests for the Checkpoint class."""

import pathlib
import numpy as np
import pytest

from aegis_sim.checkpoint import Checkpoint
from aegis_sim.dataclasses.population import Population


def _make_dummy_checkpoint(step=50):
    """Create a minimal Checkpoint with fake data for unit testing."""
    return Checkpoint(
        population=None,
        eggs=None,
        step=step,
        rng_state=np.random.default_rng(42).bit_generator.state,
        random_seed=42,
        legacy_rng_state=np.random.get_state(),
        envdrift_map=None,
        predator_population_size=0.0,
        resource_capacity=1000.0,
        final_config={"STEPS_PER_SIMULATION": 100},
        custom_config_path=pathlib.Path("dummy.yml"),
    )


def test_save_and_load(tmp_path):
    """Checkpoint round-trips through save/load."""
    cp = _make_dummy_checkpoint(step=42)
    path = tmp_path / "checkpoint"
    cp.save(path)

    loaded = Checkpoint.load(path)
    assert loaded.step == 42
    assert loaded.random_seed == 42
    assert loaded.resource_capacity == 1000.0


def test_save_is_atomic(tmp_path):
    """After save, no .tmp files should remain."""
    cp = _make_dummy_checkpoint()
    path = tmp_path / "checkpoint"
    cp.save(path)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0
    assert path.exists()


def test_load_wrong_type(tmp_path):
    """Loading a non-Checkpoint pickle should raise TypeError."""
    import pickle
    path = tmp_path / "bad_checkpoint"
    with open(path, "wb") as f:
        pickle.dump({"not": "a checkpoint"}, f)

    with pytest.raises(TypeError, match="Expected Checkpoint"):
        Checkpoint.load(path)


def test_find_latest_missing(tmp_path):
    """find_latest on a dir with no checkpoint should raise."""
    with pytest.raises(FileNotFoundError):
        Checkpoint.find_latest(tmp_path)


def test_find_latest_exists(tmp_path):
    """find_latest returns the checkpoint path when it exists."""
    cp = _make_dummy_checkpoint()
    cp.save(tmp_path / "checkpoint")

    found = Checkpoint.find_latest(tmp_path)
    assert found == tmp_path / "checkpoint"
