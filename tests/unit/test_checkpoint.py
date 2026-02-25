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


def test_save_creates_backup(tmp_path):
    """Second save should create a .bak of the first checkpoint."""
    path = tmp_path / "checkpoint"
    backup_path = path.with_suffix(".bak")

    cp1 = _make_dummy_checkpoint(step=10)
    cp1.save(path)
    assert not backup_path.exists()

    cp2 = _make_dummy_checkpoint(step=20)
    cp2.save(path)
    assert backup_path.exists()

    # Primary should be step 20, backup should be step 10
    loaded_primary = Checkpoint.load(path)
    loaded_backup = Checkpoint._load_single(backup_path)
    assert loaded_primary.step == 20
    assert loaded_backup.step == 10


def test_load_falls_back_to_backup(tmp_path):
    """If primary checkpoint is corrupt, load should fall back to .bak."""
    path = tmp_path / "checkpoint"
    backup_path = path.with_suffix(".bak")

    # Save a valid checkpoint, then save again so the first becomes .bak
    cp1 = _make_dummy_checkpoint(step=10)
    cp1.save(path)
    cp2 = _make_dummy_checkpoint(step=20)
    cp2.save(path)

    # Corrupt the primary
    with open(path, "wb") as f:
        f.write(b"truncated garbage")

    loaded = Checkpoint.load(path)
    assert loaded.step == 10


def test_load_corrupt_no_backup_raises(tmp_path):
    """If primary is corrupt and no backup exists, load should raise."""
    path = tmp_path / "checkpoint"

    # Write a corrupt file directly (no backup)
    with open(path, "wb") as f:
        f.write(b"truncated garbage")

    with pytest.raises(Exception):
        Checkpoint.load(path)


def test_save_does_not_backup_corrupt_file(tmp_path):
    """If the existing checkpoint is corrupt, save should discard it, not promote to .bak."""
    path = tmp_path / "checkpoint"
    backup_path = path.with_suffix(".bak")

    # Write a corrupt primary checkpoint
    with open(path, "wb") as f:
        f.write(b"truncated garbage")

    # Save a new valid checkpoint
    cp = _make_dummy_checkpoint(step=30)
    cp.save(path)

    # Primary should be valid step 30
    loaded = Checkpoint.load(path)
    assert loaded.step == 30

    # Backup should NOT exist (corrupt file was discarded)
    assert not backup_path.exists()


def test_save_preserves_good_backup_when_primary_corrupt(tmp_path):
    """If primary is corrupt but .bak is good, saving should keep the good .bak."""
    path = tmp_path / "checkpoint"
    backup_path = path.with_suffix(".bak")

    # Create a good backup manually
    cp_good = _make_dummy_checkpoint(step=10)
    cp_good.save(backup_path)

    # Write a corrupt primary
    with open(path, "wb") as f:
        f.write(b"truncated garbage")

    # Save a new checkpoint — corrupt primary should be discarded, good .bak untouched
    cp_new = _make_dummy_checkpoint(step=30)
    cp_new.save(path)

    loaded_primary = Checkpoint.load(path)
    assert loaded_primary.step == 30

    # The good backup should still be there (not overwritten by the corrupt file)
    loaded_backup = Checkpoint._load_single(backup_path)
    assert loaded_backup.step == 10


def test_find_latest_falls_back_to_backup(tmp_path):
    """find_latest should return .bak path if primary is missing."""
    backup_path = tmp_path / "checkpoint.bak"
    cp = _make_dummy_checkpoint(step=10)
    cp.save(backup_path)

    found = Checkpoint.find_latest(tmp_path)
    assert found == backup_path
