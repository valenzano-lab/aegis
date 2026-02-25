"""Test seed mode (starting a new sim from a pickled population)."""

import pickle
import aegis_sim
from aegis_sim.dataclasses.population import Population


def test_seed_from_pickle(base_config, write_config, tmp_path):
    """Seeding from a pickle should start a fresh sim with that population."""
    # First run: produce a pickle
    config_with_pickle = {**base_config, "PICKLE_RATE": 25}
    config_path = write_config(config_with_pickle, name="seed_source")
    odir = tmp_path / "seed_source"

    aegis_sim.run(
        custom_config_path=config_path,
        pickle_path=None,
        overwrite=False,
        custom_input_params={},
    )

    # Find a pickle file (files are named by step number, no extension)
    pickle_files = list(odir.glob("pickles/*"))
    pickle_files = [f for f in pickle_files if f.is_file()]
    assert len(pickle_files) > 0, "No pickle files produced"
    pickle_path = pickle_files[0]

    # Second run: seed from pickle
    config_path2 = write_config(base_config, name="seeded_sim")
    odir2 = tmp_path / "seeded_sim"

    aegis_sim.run(
        custom_config_path=config_path2,
        pickle_path=pickle_path,
        overwrite=False,
        custom_input_params={},
    )

    assert odir2.exists()
    # Should have full STEPS_PER_SIMULATION lines (fresh run, not a resume)
    popsize_file = odir2 / "popsize_before_reproduction.csv"
    lines = popsize_file.read_text().strip().splitlines()
    assert len(lines) == base_config["STEPS_PER_SIMULATION"]
