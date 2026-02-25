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

## Notes

- Unit tests are fast and don't run simulations.
- Functional tests run short simulations (100 steps) and verify outputs.
- New tests use pytest's `tmp_path` fixture so output is cleaned up automatically.
- Legacy tests write output to `tests/functional/experiments/` which is gitignored.
- `test_zcontainer.py` depends on experiment output from other functional tests,
  so it should run after them (the `z` prefix ensures alphabetical ordering).

## What's still missing

See `tests/TODO.md` for the full prioritized list. Summary of gaps:

- `Phenotypes` — clip_array_to_01, extract(), gaussian_smoothing()
- `ParameterManager` — init_from_config, read_config_file, validate()
- `Trait` — construction and validation
- `Resources` — replenish, reduce, scavenge
- `Starvation` — get_mask_kill (deficit mode, consecutive mode, max cap)
- `MatingManager` — pair_up_polygamously
- `SexSystem` — get_sex
- `Mutator._mutate_by_index` — the fast-path mutation method
- `Infection.__call__` — full SIR step
- `CompositeArchitecture` / `ModifyingArchitecture` — coupled to parameterization; needs integration fixture
- `GPM` / `GPM_decoder` — coupled to parameterization
- `PopgenStats` — theta_w, theta_pi, tajimas_d, SFS, Fay & Wu (sample-based stats)
- `utilities/analysis/*` — survival, reproduction, leslie, genome analysis functions
- Functional tests — most `test_zcontainer.py` methods still lack data correctness assertions
