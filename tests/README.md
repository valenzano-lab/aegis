# Tests for AEGIS

## Setup

```bash
pip install -e ".[dev]"
```

## Running

```bash
# Run everything
pytest tests/ -v

# Unit tests only (fast, no simulation runs)
pytest tests/unit/ -v

# Functional tests only (runs actual simulations)
pytest tests/functional/ -v

# Functional tests in parallel (uses all CPU cores)
pytest tests/functional/ -n auto -v

# A specific test
pytest tests/functional/test_checkpoint_resume.py::test_resume_no_duplicate_rows -v

# Stop on first failure
pytest tests/ -x -v
```

## Structure

```
tests/
├── conftest.py                          # shared fixtures (configs, write helper)
├── unit/                                # fast, isolated tests
│   ├── test_checkpoint.py               # Checkpoint save/load/find
│   └── test_truncation.py               # _count_recordings, _truncate_file
└── functional/                          # end-to-end simulation tests
    ├── conftest.py                      # shared experiment path for legacy tests
    ├── test_basic_sim.py                # fresh sim output files and line counts
    ├── test_checkpoint_resume.py        # checkpoint + resume, no duplicate data
    ├── test_seed_mode.py                # seed from pickle
    ├── test_cli.py                      # CLI arg parsing and validation
    ├── test_sim.py                      # basic sim smoke test
    ├── test_frailty.py                  # frailty modifier parameter sweep
    ├── test_maturity.py                 # maturation age parameter sweep
    ├── test_phenotypes.py               # genome size and bits per locus
    ├── test_resources.py                # starvation parameters
    ├── test_gui.py                      # Dash app serves correctly
    └── test_zcontainer.py               # Container class reads sim output
```

## Notes

- Unit tests are fast and don't run simulations.
- Functional tests run short simulations (100 steps) and verify outputs.
- New tests use pytest's `tmp_path` fixture so output is cleaned up automatically.
- Legacy tests write output to `tests/functional/experiments/` which is gitignored.
- `test_zcontainer.py` depends on experiment output from other functional tests,
  so it should run after them (the `z` prefix ensures alphabetical ordering).
