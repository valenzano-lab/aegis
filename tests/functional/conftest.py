"""Shared state for legacy functional tests."""

import pathlib

# Shared experiment output directory — used by tests that write configs and run sims
test_experiment_path = pathlib.Path(__file__).absolute().parent / "experiments"
test_experiment_path.mkdir(exist_ok=True, parents=True)
