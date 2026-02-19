"""Test that output_summary.json reports the extinct flag correctly."""

import json
import aegis_sim


def test_surviving_population_not_marked_extinct(base_config, write_config, tmp_path):
    """A simulation that runs to completion with population alive should report extinct=false."""
    # Use a large initial population and short run to ensure survival
    config = {
        **base_config,
        "STEPS_PER_SIMULATION": 10,
        "INITIAL_POPULATION_SIZE": 1000,
    }
    config_path = write_config(config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    summary = json.loads((odir / "output_summary.json").read_text())
    assert summary["extinct"] is False, (
        f"Population survived all steps but extinct={summary['extinct']}"
    )


def test_extinct_population_marked_extinct(base_config, write_config, tmp_path):
    """A simulation where the population dies should report extinct=true."""
    # Tiny population + short age limit + high maturation age → no reproduction, guaranteed extinction
    config = {
        **base_config,
        "STEPS_PER_SIMULATION": 500,
        "INITIAL_POPULATION_SIZE": 2,
        "AGE_LIMIT": 5,
        "MATURATION_AGE": 100,  # can never reach maturity before dying
    }
    config_path = write_config(config)
    odir = tmp_path / config_path.stem

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    summary = json.loads((odir / "output_summary.json").read_text())
    assert summary["extinct"] is True, (
        f"Population should have gone extinct but extinct={summary['extinct']}"
    )
