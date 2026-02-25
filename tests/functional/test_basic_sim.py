"""Test that a fresh simulation produces expected output files."""

import aegis_sim


def test_basic_sim_creates_output_files(base_config, write_config, tmp_path):
    """Run a short sim and verify key output files exist."""
    config_path = write_config(base_config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    assert odir.exists()
    assert (odir / "popsize_before_reproduction.csv").exists()
    assert (odir / "popsize_after_reproduction.csv").exists()
    assert (odir / "progress.log").exists()


def test_popsize_line_count(base_config, write_config, tmp_path):
    """Per-step files should have exactly STEPS_PER_SIMULATION lines."""
    steps = base_config["STEPS_PER_SIMULATION"]
    config_path = write_config(base_config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    popsize_file = odir / "popsize_before_reproduction.csv"
    lines = popsize_file.read_text().strip().splitlines()
    assert len(lines) == steps, f"Expected {steps} lines, got {len(lines)}"


def test_progress_log_line_count(base_config, write_config, tmp_path):
    """progress.log should have 1 header + expected number of data lines."""
    steps = base_config["STEPS_PER_SIMULATION"]
    rate = base_config["LOGGING_RATE"]
    config_path = write_config(base_config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    progress_file = odir / "progress.log"
    lines = progress_file.read_text().strip().splitlines()
    # 1 header + recordings at step 1 and every LOGGING_RATE steps
    expected_data = 1 + steps // rate  # step 1 + multiples of rate in [1, steps]
    expected_total = 1 + expected_data  # header + data
    assert len(lines) == expected_total, f"Expected {expected_total} lines, got {len(lines)}"


def test_overwrite_flag(base_config, write_config, tmp_path):
    """Running twice without --overwrite should raise; with it should succeed."""
    config_path = write_config(base_config)

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    # Second run without overwrite should fail
    import pytest
    with pytest.raises(Exception, match="already exists"):
        aegis_sim.run(
            custom_config_path=config_path,
            pickle_path=None,
            overwrite=False,
            custom_input_params={},
        )

    # With overwrite should succeed
    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=True,
        custom_input_params={},
    )
