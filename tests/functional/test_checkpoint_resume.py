"""Test checkpoint save, load, and resume functionality."""

import aegis_sim
from aegis_sim.checkpoint import Checkpoint


def test_checkpoint_file_created(checkpoint_config, write_config, tmp_path):
    """A sim with CHECKPOINT_RATE > 0 should produce a checkpoint file."""
    config_path = write_config(checkpoint_config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    assert (odir / "checkpoint").exists()


def test_checkpoint_load_roundtrip(checkpoint_config, write_config, tmp_path):
    """Load a checkpoint and verify its fields are populated."""
    config_path = write_config(checkpoint_config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    cp = Checkpoint.load(odir / "checkpoint")
    assert cp.step > 0
    assert cp.population is not None
    assert cp.final_config is not None
    assert cp.rng_state is not None


def test_resume_continues_from_checkpoint(checkpoint_config, write_config, tmp_path):
    """Resume should pick up from the checkpoint step and finish the sim."""
    steps = checkpoint_config["STEPS_PER_SIMULATION"]
    config_path = write_config(checkpoint_config)
    odir = tmp_path / config_path.stem

    # Run full sim to create checkpoint
    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    # Record the checkpoint step
    cp = Checkpoint.load(odir / "checkpoint")
    assert cp.step <= steps

    # Resume from checkpoint
    aegis_sim.run(
        custom_config_path=None,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
        resume_path=odir,
    )

    # Popsize file should still have exactly STEPS_PER_SIMULATION lines
    popsize_file = odir / "popsize_before_reproduction.csv"
    lines = popsize_file.read_text().strip().splitlines()
    assert len(lines) == steps, f"Expected {steps} lines, got {len(lines)}"


def test_resume_no_duplicate_rows(checkpoint_config, write_config, tmp_path):
    """After resume, per-step files should not have more lines than STEPS_PER_SIMULATION."""
    steps = checkpoint_config["STEPS_PER_SIMULATION"]
    config_path = write_config(checkpoint_config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    # Resume (simulates crash-and-restart scenario)
    aegis_sim.run(
        custom_config_path=None,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
        resume_path=odir,
    )

    per_step_files = [
        "popsize_before_reproduction.csv",
        "popsize_after_reproduction.csv",
        "eggnum_after_reproduction.csv",
    ]
    for fname in per_step_files:
        f = odir / fname
        if f.exists():
            lines = f.read_text().strip().splitlines()
            assert len(lines) == steps, (
                f"{fname}: expected {steps} lines, got {len(lines)}"
            )


def test_resume_rate_based_files_no_duplicates(checkpoint_config, write_config, tmp_path):
    """Rate-based files should have correct line counts after resume."""
    steps = checkpoint_config["STEPS_PER_SIMULATION"]
    rate = checkpoint_config["INTERVAL_RATE"]
    config_path = write_config(checkpoint_config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    aegis_sim.run(
        custom_config_path=None,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
        resume_path=odir,
    )

    # genotypes.csv: 2 header lines + data at INTERVAL_RATE
    genotypes = odir / "gui" / "genotypes.csv"
    if genotypes.exists():
        lines = genotypes.read_text().strip().splitlines()
        expected_data = 1 + steps // rate  # step 1 + multiples of rate
        expected_total = 2 + expected_data
        assert len(lines) == expected_total, (
            f"genotypes.csv: expected {expected_total} lines, got {len(lines)}"
        )


def test_checkpoint_not_found_raises(tmp_path):
    """Resuming from a directory with no checkpoint should raise FileNotFoundError."""
    import pytest
    empty_dir = tmp_path / "no_checkpoint"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        Checkpoint.find_latest(empty_dir)
