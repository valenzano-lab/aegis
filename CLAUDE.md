# AEGIS — Project Context for Claude Code

## What this project is

**AEGIS** (Aging of Evolving Genomes In Silico) is a Python evolutionary genetics simulation engine for life history evolution in age-structured populations. It is used by the Valenzano Lab for research on aging and life history theory.

- PyPI package: `aegis-sim` (v2.3.0.2)
- GitHub: https://github.com/valenzano-lab/aegis
- Active development branch: `v2`
- Authors: Martin Bagic (lead developer v2), Dario Valenzano (PI, v1 + v2)

## Package layout

```
src/
  aegis/         — CLI entry point and argument parsing
  aegis_sim/     — core simulation engine (the thing we mostly work on)
  aegis_gui/     — Dash web GUI (lower priority for research changes)
tests/
  unit/          — isolated unit tests per module
  functional/    — end-to-end simulation tests
```

## How to install for development

```bash
pip install -e ".[dev]"   # from repo root
```

## How to run

```bash
aegis sim -c path/to/config.yml       # fresh simulation
aegis sim -c config.yml -r            # resume from checkpoint
aegis sim -c config.yml -r --extend N # resume and extend to N steps
aegis gui                             # launch Dash GUI
```

Config files are YAML. Key parameters: `RANDOM_SEED`, `STEPS_PER_SIMULATION`, `AGE_LIMIT`, `INITIAL_POPULATION_SIZE`.

## Core simulation loop

`Bioreactor.run_step()` in `src/aegis_sim/bioreactor.py`:
1. Check extinction
2. Apply mortalities in `MORTALITY_ORDER` (intrinsic, abiotic, infection, predation, starvation)
3. Replenish resources
4. `growth()` — increment individual sizes (currently a stub: `sizes += 1`)
5. `reproduction()` — generate eggs, apply carrying capacity
6. `age()` — increment ages, kill those hitting `AGE_LIMIT`
7. `hatch()` — turn eggs into living individuals (respects `INCUBATION_PERIOD`)
8. Record everything

## Population data model

`Population` in `src/aegis_sim/dataclasses/population.py` — all attributes are NumPy arrays of equal length:

| Attribute | dtype | Meaning |
|-----------|-------|---------|
| `genomes` | Genomes | binary genome array |
| `ages` | int32 | age in simulation steps |
| `births` | int32 | cumulative offspring produced |
| `birthdays` | int32 | step at which individual was born/hatched |
| `phenotypes` | Phenotypes | derived from genomes via `architect` |
| `infection` | int32 | infection status (0=healthy, -1=dead, etc.) |
| `sizes` | float32 | body size (currently stub) |
| `sexes` | — | sex assignment |
| `generations` | — | **always None** — not yet implemented |

## Genetic traits

5 traits encoded in genomes (`src/aegis_sim/constants.py`):
- `surv` — survival probability (age-specific)
- `repr` — reproduction probability (age-specific)
- `muta` — mutation rate
- `neut` — neutral (no phenotypic effect)
- `grow` — growth rate (stub, not used in current loop)

Two genetic architectures (`GENARCH_TYPE`):
- `composite` — bits per locus map directly to trait × age matrix
- `modifying` — a phenomap defines pleiotropic effects of loci

## Submodel singletons

Some submodels are class instances, others use a module-as-object pattern (`init(self, ...)` called on the module). Both patterns coexist:

| Submodel | Pattern |
|----------|---------|
| `abiotic`, `predation`, `infection`, `reproduction`, `sexsystem`, `matingmanager`, `architect` | Class instances on the `submodels` module |
| `resources`, `starvation`, `frailty`, `mutator`, `ploider` | Module-level singletons initialized via `init(self, ...)` |

## Known issues / technical debt

### Dual RNG (important for reproducibility)
Two RNGs are used simultaneously — `np.random` (legacy global) and `np.random.default_rng` (modern). Both are seeded from `RANDOM_SEED`. Any refactor that adds or removes a call in one stream shifts all subsequent random numbers. Documented in `variables.py`.

Submodels using **legacy** `np.random`: `population.initialize`, `matingmanager`, `abiotic`, `envdrift`, `recombination`, `gpm_decoder`, `popgenstats`.

### Stub / incomplete features
- `growth()` in `bioreactor.py` just does `sizes += 1` — resource-based growth is commented out
- `generations` field in `Population` is set to `None` everywhere — never tracked
- `parental_generations` in `reproduction()` uses `np.zeros(...)` (acknowledged as wrong)
- `CARRYING_CAPACITY_EGGS` slicing takes the *last* N eggs (positional bias) instead of random sample
- `STARVATION_RESPONSE` parameter is fully commented out — starvation refactor is incomplete

### Broken tests
- `tests/unit/test_checkpoint.py` — fails with `ModuleNotFoundError` (installed version vs workspace mismatch)
- `tests/unit/test_truncation.py` — 11 tests fail for same reason

### Missing unit tests
See `tests/TODO.md` for the full list. Priority 1 gaps:
- `phenotypes.py`, `parametermanager.py`, `resources.py`, `starvation.py`, `matingmanager.py`, `sexsystem.py`, full infection SIR step

## How to run tests

```bash
pytest tests/              # all tests
pytest tests/unit/         # unit tests only
pytest tests/functional/   # functional tests only
pytest -x                  # stop on first failure
pytest -n auto             # parallel (requires pytest-xdist)
```

## Research context

This is a **research tool** used by the Valenzano Lab. Changes are driven by scientific requirements, not software product requirements. When adding features:
- Prefer scientific accuracy over API elegance
- New parameters should follow the `Parameter(...)` pattern in `default_parameters.py`
- Config files are YAML; defaults must be sensible for typical runs
- Output files land in a directory named after the config file (e.g., `config.yml` → `config/`)

## Branch strategy

- `v2` is the main/default branch
- Feature work and research-driven changes should be developed on topic branches and merged via PR
- The lab may run long simulations — checkpoint compatibility matters; avoid breaking the checkpoint format without a migration plan
