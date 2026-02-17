# Unit Test Coverage TODO

## What's covered

| Module | Test file | What's tested |
|--------|-----------|---------------|
| `dataclasses/genomes.py` | `test_genomes.py` | init, len, flatten, get, getitem, add, keep, shape, get_array |
| `dataclasses/population.py` | `test_population.py` | init validation, len, getitem, imul, iadd, pickle, make_eggs (skip) |
| `parameterization/parameter.py` | `test_parameter.py` | convert, valid, validate_dtype, validate_inrange, get_name |
| `submodels/abiotic.py` | `test_abiotic.py` | all 7 waveforms, offset, invalid shape |
| `submodels/frailty.py` | `test_frailty.py` | modify() formula at various ages |
| `submodels/predation.py` | `test_predation.py` | init, zero prey, fraction bounds, Verhulst growth |
| `submodels/infection.py` | `test_infection.py` | get_infection_probability formula |
| `submodels/genetics/ploider.py` | `test_ploider.py` | init validation, diploid_to_haploid (homo/hetero/batch) |
| `submodels/reproduction/mutation.py` | `test_mutation.py` | init, mutate_by_bit, age multiplier |
| `submodels/reproduction/recombination.py` | `test_recombination.py` | zero rate, shape, bit conservation |
| `recording/ticker.py` | `test_ticker.py` | write, read, since_last, has_stopped |
| `recording/progressrecorder.py` | `test_progress_recorder.py` | get_dhm, init_headers, write_to_progress_log, write (patched) |
| `recording/recordingmanager.py` | `test_recording_manager.py` | make_odir, _count_recordings, _truncate_file |
| `recording/popsizerecorder.py` | `test_popsize_recorder.py` | write, write_egg_num_after_reproduction |
| `recording/configrecorder.py` | `test_config_recorder.py` | write_final_config_file YAML round-trip |
| `recording/resourcerecorder.py` | `test_resource_recorder.py` | write_before/after_scavenging (patched) |
| `recording/simpleprogressrecorder.py` | `test_simple_progress_recorder.py` | check_when_last_updated |
| `utilities/popgenstats.py` | `test_popgenstats.py` | harmonic, harmonic_sq, make_3D/4D, reference genome, segregating sites, calc (haploid/diploid heterozygosity, theta) |
| `checkpoint.py` | `test_checkpoint.py` | save, load, find_latest (pre-existing) |

## Functional test improvements done

| File | Change |
|------|--------|
| `test_zcontainer.py` | Added real assertions: data types, row counts, value ranges, config content verification. Previously smoke-test only. |

## What's missing

### Priority 1 — Core simulation logic

- [ ] `dataclasses/phenotypes.py` — clip_array_to_01, extract(), get_trait_position(), gaussian_smoothing()
- [ ] `parameterization/trait.py` — Trait construction, _validate() (evolvable/agespecific combos, interpreter types)
- [ ] `parameterization/parametermanager.py` — init_from_config, read_config_file, validate()
- [ ] `submodels/resources/resources.py` — Resources.replenish, reduce, scavenge
- [ ] `submodels/resources/starvation.py` — Starvation.get_mask_kill (deficit mode, consecutive mode, frailty interaction, max cap)
- [ ] `submodels/reproduction/matingmanager.py` — pair_up_polygamously
- [ ] `submodels/reproduction/sexsystem.py` — get_sex (50/50 distribution)
- [ ] `submodels/reproduction/mutation.py` — _mutate_by_index (the fast path, currently untested)
- [ ] `submodels/infection.py` — __call__ (full SIR step: infection, recovery, fatality)

### Priority 2 — Genetics internals

- [ ] `submodels/genetics/composite/architecture.py` — get_number_of_bits, get_shape, init_genome_array, compute
- [ ] `submodels/genetics/composite/interpreter.py` — Interpreter.call (uniform, exp, binary, switch, threshold, etc.)
- [ ] `submodels/genetics/modifying/architecture.py` — get_number_of_bits, get_shape, init_genome_array, compute
- [ ] `submodels/genetics/modifying/gpm.py` — GPM.phenodiff, __call__
- [ ] `submodels/genetics/modifying/gpm_decoder.py` — GPM_decoder.get_total_phenolist, Genblock.get_phenolist
- [ ] `submodels/genetics/envdrift.py` — Envdrift.will_evolve, evolve, call
- [ ] `submodels/genetics/architect.py` — Architect.__call__ (end-to-end genomes → phenotypes)

### Priority 3 — PopgenStats (remaining)

- [ ] `utilities/popgenstats.py` — theta_w, theta_pi, tajimas_d with known inputs (need sample-based tests)
- [ ] `utilities/popgenstats.py` — get_sfs, get_theta_h, get_fayandwu_h
- [ ] `utilities/popgenstats.py` — get_genotype_frequencies (diploid)
- [ ] `utilities/popgenstats.py` — get_genomes_sample (sampling logic)

### Priority 4 — Utilities and analysis

- [ ] `utilities/analysis/survival.py` — get_mortality, get_survivorship, get_life_expectancy, get_longevity
- [ ] `utilities/analysis/reproduction.py` — get_fertility, get_cumulative_reproduction, get_lifetime_reproduction
- [ ] `utilities/analysis/leslie.py` — leslie_matrix, leslie_breakdown
- [ ] `utilities/analysis/genome.py` — get_sorted_allele_frequencies, get_derived_allele_freq
- [ ] `utilities/get_folder_size.py` — get_folder_size_with_du, convert_size, get_folder_size_python
- [ ] `utilities/funcs.py` — skip() (needs variables/parametermanager mocking)

### Priority 5 — Functional test improvements (remaining)

- [ ] `test_zcontainer.py` — verify popgenstats output files exist and contain valid data
- [ ] `test_zcontainer.py` — verify feather file column structure matches expected traits
- [ ] `test_basic_sim.py` — verify CSV content (not just line counts)
- [ ] `test_sim.py` / `test_frailty.py` / `test_maturity.py` — add assertions on output values, not just "doesn't crash"

### Not worth unit testing

- `constants.py` — just static tuples
- `variables.py` — module-level state, tested implicitly
- `recording/recorder.py` — 2-line base class
- `bioreactor.py` — tightly coupled; covered by functional tests
- `recording/featherrecorder.py` — thin wrapper around pandas; covered by functional tests
- `recording/intervalrecorder.py` — thin wrapper; covered by functional tests
- `recording/checkpointrecorder.py` — covered by test_checkpoint.py
- `recording/summaryrecorder.py` — reads other recorders' output; covered by functional tests
- `aegis_gui/` — Dash GUI components (need dash.testing framework, separate effort)

## Pre-existing test issues

- [ ] `test_checkpoint.py` — import fails (`ModuleNotFoundError: No module named 'aegis_sim.checkpoint'`); likely stale installed version vs workspace mismatch
- [ ] `test_truncation.py` — all 11 tests fail (same version mismatch); superseded by `test_recording_manager.py`
