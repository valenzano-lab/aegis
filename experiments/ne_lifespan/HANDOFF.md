# Ne → lifespan experiment — handoff

**Question (Ruchitha):** is the generational lifespan reduction driven by carrying capacity K or by Ne?
**Design (Dario):** pre-evolve ONE population → subsample to a range of Ne → run each forward at
CONSTANT K=N (birth-regulated, no starvation) → re-measure Ne at the end and relate realized Ne to
evolved lifespan. Common ancestor ⇒ forward divergence is attributable to Ne alone.

This directory holds the validated pipeline. It was developed in a throwaway session scratchpad
(now gone); run outputs were not copied (cheap to regenerate). Scripts only.

## Scripts

**Local / pilot** (single process, needs an aegis env on this machine):
- `preevolve.py K_BURN BURN_STEPS FWD_STEPS N1 N2 ...` — burn-in, subsample to each N, forward at K=N.
- `run_sim.py out.yml key=val ...` — single run via `aegis_sim.run` (MA/asexual base, no GUI).
- `ne_sweep.sh` — from-scratch K sweep (superseded by the pre-evolve design; kept for reference).

**Cluster (gen100/SGE)** — same design as `preevolve.py`, split so it maps onto an array:
- `ne_lifespan_configs.py --outdir DIR` — writes `burn_s{seed}.yml` + `fwd_N{N}_s{seed}.yml`, prints the
  exact `qsub -t` ranges. N is zero-padded so the array runs ascending in N (cheap arms at low task IDs).
- `subsample.py BURN_PICKLE N OUT_PKL --rng-seed S` — the common-ancestor draw; deterministic in (seed, N).
- `ne_lifespan_qsub.sh` — `PHASE=1` burn-in per seed, `PHASE=2` subsample+forward per (N, seed).

**Fragmentation calibration (cluster)** — gate on the fragmentation arm, see below:
- `lattice_calibration_configs.py --outdir DIR` — 6 arms, identical K, viscosity varied.
- `lattice_calibration_qsub.sh` — plain SGE array, no phases; aborts if the lattice never engaged.
- `analyze_lattice_calibration.py <rundir> ...` — N-vs-K, isolation-by-distance index, global Ne.
  **Stdlib-only**, so it runs without an aegis env; `--selftest` validates the metric.

**Analysis** (needs pandas+pyarrow, not the full engine):
- `analyze_contrast.py <rundir> ...` — per run: N̄, Ne_demog (trough-driven), **Ne_genetic (units-correct)**, evolved e0.
- `runs/genetic_ne.py <rundir> ...` — realized Ne table + N→Ne plot; `--selftest` is stdlib-only.

## VALIDATED FACTS (do not re-derive)
- **Genetic-Ne units trap (critical).** In AEGIS `theta_w` is GENOME-TOTAL (S/a_n, S not divided by
  sites) but `theta` (=2·ploidy·ne·µ) is PER-SITE. They differ by a factor **L = n_loci×bits_per_locus**.
  Correct estimator: **Ne_genetic = θ_w/(L·2·ploidy·µ) = ne·θ_w/(θ·L)**. Validated on real output
  (θ_w/θ ≈ L=800; both forms return the same Ne ≈ census at equilibrium). `analyze_contrast.py` has
  the correct form. `runs/genetic_ne.py` **FIXED 2026-08-15** — it now reads L from
  `popgen/allele_frequencies.csv` (row length = ploidy·L), applies `/L`, and cross-checks the ratio
  form against `θ_w/(L·2·ploidy·µ)` using the recorded µ. If `allele_frequencies.csv` is absent it
  reports demographic Ne only rather than a number wrong by ~800×.
- **These runs are DIPLOID.** `PLOIDY` defaults to 2 and `ne_ma_ap_configs.build()` never sets it, so
  "asexual" here means clonal diploid (RECOMBINATION_RATE=0), not haploid. The genetic-Ne math is
  nonetheless correct as written: `get_genomes_sample()` unfolds the sample along the chromosome axis,
  so `nsample` counts CHROMOSOMES and `a_n = harmonic(nsample-1)` is the right Watterson denominator;
  `segregating_sites_gsample` is counted with ploidy=1 over that unfolded array (i.e. over L); and
  `theta = 2·ploidy·ne·µ = 4·ne·µ` is the standard diploid per-site θ. Consequence to remember:
  **`POPGENSTATS_SAMPLE_SIZE=100` samples 100 chromosomes ≈ 50 individuals.**
- **`MAX_OFFSPRING_NUMBER` defaults to 1** ⇒ the population cannot boom-bust ⇒ no real oscillations ⇒
  Ne≈N regardless of starvation. Need MAX_OFFSPRING_NUMBER>1 for any demographic-oscillation effect.
- **Brief troughs do NOT crater Ne.** Ne_demog = harmonic mean of per-step census is dominated by TIME
  SPENT at low N, not depth touched. Troughs to 115 for a few steps leave Ne_demog ≈ 275. Sustained
  bottlenecks (slow recovery) lower Ne but also drag N̄ down.
- **`get_ne()` is unreliable twice over:** it samples census only every POPGENSTATS_RATE steps (aliases
  troughs → overestimates Ne) and ignores reproductive-skew. Use per-step `popsize_after_reproduction.csv`.
- **Density-regulation MODE strongly affects evolved lifespan via the EXTRINSIC-MORTALITY channel**
  (starvation-death → steeper force-of-selection decline → faster aging), NOT via Ne. Every
  starvation regime evolved e0≈15–18; every birth-regulated regime e0≈19–20 at matched K. To ISOLATE
  Ne you MUST hold the regulation mode fixed (birth-regulated) — which the pre-evolve design does.
- **`sim()` starts the ticker via multiprocessing (spawn on macOS)** → any driver script MUST guard
  execution under `if __name__ == "__main__":` or the child re-runs and rmtree's outputs.
- Read aging off the **genetic `surv` phenotype** (snapshots/phenotypes), never realized deaths.
  e0 = Σ cumprod(mean surv per age).

## INTERIM RESULT (single seed, small range — direction only)
Pre-evolved common ancestor, forward at constant K, birth-regulated MA:
| N=K | Ne_genetic (realized) | evolved e0 |
|--:|--:|--:|
| 100 | 131 | 18.83 |
| 250 | 150 | 19.07 |
| 500 | 166 | 20.24 |
Higher Ne → longer life (the load story), a trend INVISIBLE in from-scratch runs (noise). Range too
small and n=1 to conclude.

## NEXT
1. ~~Fix `runs/genetic_ne.py`~~ **DONE 2026-08-15** (see the units trap above). It now needs
   `POPGENSTATS_RATE>0` *and* `popgen/allele_frequencies.csv` (to read L); both configs provide them.
2. ~~SGE wrapper~~ **WRITTEN 2026-08-15, NOT YET RUN.** Launch on Merlin:
   ```
   CONFIG_DIR=/wins/vlzno/projects/aegis_ne_lifespan
   python experiments/ne_lifespan/ne_lifespan_configs.py --outdir $CONFIG_DIR
   mkdir -p logs
   CONFIG_DIR=$CONFIG_DIR PHASE=1 qsub -t 1-3  experiments/ne_lifespan/ne_lifespan_qsub.sh
   # CHECK the burn-ins reached K and equilibrated, THEN:
   CONFIG_DIR=$CONFIG_DIR PHASE=2 qsub -t 1-15 experiments/ne_lifespan/ne_lifespan_qsub.sh
   ```
   Defaults: K_BURN=10000, N∈[100,316,1000,3162,10000], seeds 1–3, 200k burn-in + 200k forward steps.
   **Step counts are a guess and are the thing to sanity-check first** — the forward phase must be long
   enough for mutation load to accumulate at the LOW-Ne end, and 200k has not been calibrated for that.
3. Report realized-Ne vs e0 across the full range for Ruchitha.

## FRAGMENTATION ARM (added 2026-08-15) — the way to get Ne range at CONSTANT K

**Correcting an earlier claim in this file's history:** "equal K + Ne spanning 10²–10⁴ is impossible"
is true only for a WELL-MIXED population (binomial reproduction cannot be overdispersed, so Ne is
pinned within ~2× of N). **Spatial structure breaks that.** Under `LATTICE_MODE`, mating and offspring
placement are local, so drift is governed by Wright's *neighbourhood* size, not global census — which
can sit orders of magnitude below N. Knobs: `MIGRATION_RATE` (default 0.1, "the dominant viscosity
knob"), `MIGRATION_LONG_RATE` (default 0.005, set 0 for pure isolation-by-distance),
`LATTICE_TARGET_DENSITY` (0.3 baseline). Asexual lattice runs already exist in `runs/`.

This gives Dario's original Step 2 — identical K, identical census N, identical N·u, wide Ne range —
and it *is* population fragmentation, so design and biology coincide.

### VALIDATED FACTS about lattice mode (verified by reading the engine, 2026-08-15)
- ⚠️ **SILENT FAILURE.** `lattice.assign_initial_positions()` is called ONLY from
  `Population.initialize()` — never on a pickled population. Every lattice path in the bioreactor is
  guarded by `positions is not None`. So **a lattice run seeded from a non-lattice pickle runs
  WELL-MIXED while its config says `LATTICE_MODE=True`**, with no error. ⇒ the fragmentation burn-in
  must ITSELF be in lattice mode. Both the qsub and the analyzer now check for lattice output and
  fail loudly; do not remove those checks.
- `LATTICE_MODE`, `MIGRATION_RATE`, `MIGRATION_LONG_RATE` are NOT in `STRUCTURAL_PARAMETERS`, so they
  **can** be `--override`n on resume — that is what lets all arms branch off one ancestor.
- `resync_occupancy_from_positions()` rebuilds occupancy from `population.positions` every step, and
  the checkpoint pickles the population, so positions survive resume.
- Lattice mode adds LOCAL density regulation (offspring go to a random adjacent empty cell; **birth
  fails if none is free**) on top of `REPRODUCTION_REGULATION`. Different mechanism from the rest of
  the experiment — this is the main thing calibration must check.
- ⚠️ **The panmictic Ne estimator does not apply.** Under fragmentation, pooling demes inflates
  apparent diversity (Wahlund), so global θ_w can hold steady or RISE while local selection efficiency
  collapses. `runs/genetic_ne.py` on a global sample would report "Ne unchanged" and be wrong.
  Flat global Ne + rising structure is the *signature*, not a null result.

### CALIBRATION RESULT (RAN 2026-08-15, job 974216, K=3000, 5000 steps, seed 1)
| arm | migration | long | N_mean | occ | **F_ST** | Ne_glob |
|---|--:|--:|--:|--:|--:|--:|
| cal_1 wellmixed (no lattice) | — | — | 3000 | — | — | 251 |
| cal_2 | 0.5 | 0 | 3000 | 0.30 | 0.245 | 210 |
| cal_3 | 0.1 | 0 | 3000 | 0.30 | 0.445 | 227 |
| cal_4 | 0.01 | 0 | 3000 | 0.30 | 0.677 | 266 |
| cal_5 | 0.0 | 0 | 3000 | 0.30 | 0.680 | 276 |
| cal_6 | 0.01 | 0.005 | 3000 | 0.30 | 0.172 | 199 |

- **Q1 PASS.** N_mean = N_min = 3000 and occupancy = 0.30 in EVERY arm. Lattice local density
  regulation (birth fails with no adjacent empty cell) does **not** depress N below K. Equal-K holds.
- **Q2/Q3 PASS.** F_ST is monotone in viscosity over a 4× spread. Viscosity is a real structure knob.
- **SATURATION below migration ≈ 0.01**: 0.677 (m=0.01) vs 0.680 (m=0) are the same. Do **not** spend
  an arm on m=0 — the axis is exhausted by m=0.01.
- **LONG-DISTANCE DISPERSAL IS THE STRONGER KNOB.** `MIGRATION_LONG_RATE=0.005` on top of m=0.01 gives
  F_ST 0.172 — *below* m=0.5 with no long dispersal (0.245). A rare global channel erases more
  structure than 50× more local diffusion. Local migration is diffusive and mixes slowly across the
  lattice; long dispersal is global. **Use `MIGRATION_LONG_RATE` as the primary sweep axis.**
- **No panmictic lattice reference yet.** Island-model inversion `Nm ≈ (1−F_ST)/(4·F_ST)` gives
  Nm ≈ 0.12 (cal_5) to 1.2 (cal_6): drift beats migration in *every* arm, so even cal_6 is structured.
  The sweep needs a genuinely well-mixed lattice endpoint — push `MIGRATION_LONG_RATE` toward 0.05.
- **Q4 CONFIRMED — global Ne is blind.** Ne_glob moves only 199→276 while F_ST spans 4×, and it moves
  in the *Wahlund* direction (more isolation → more retained global diversity → higher apparent Ne).
  `runs/genetic_ne.py` on a global sample MUST NOT be used to measure the fragmentation arm.
  The sweep needs within-neighbourhood sampling for local Ne.
- Sampling caveat: ~30 individuals/block inflates F_ST by roughly (1−F_ST)/2n ≈ 0.017 — negligible
  against a 0.17–0.68 signal, and common to all arms.
- Caveat on Ne_glob generally: at 5000 steps (~400 generations) diversity is far from its 4Ne·µ
  equilibrium (~2Ne ≈ 6000 generations), so all Ne_glob values are transients. F_ST is trustworthy
  because spatial structure builds on the migration timescale, not the coalescent one.

### CALIBRATION BATCH 2 RESULT (long-dispersal axis, 2026-08-15) — THE SWEEP GRID
`MIGRATION_RATE=0.01` fixed, `MIGRATION_LONG_RATE` varied. K=3000, 5000 steps, seed 1.

| long | F_ST | Nm = (1−F_ST)/4F_ST |
|--:|--:|--:|
| 0 | **0.677** | 0.12 |
| 0.001 | **0.334** | 0.50 |
| 0.005 | **0.172** | 1.21 |
| 0.02 | **0.092** | 2.47 |
| 0.05 | 0.058 | 4.09 |
| 0.1 | **0.045** | 5.27 |

- **REPRODUCIBILITY CONFIRMED.** `cal_1_ld0000` (m=0.01, long=0) returned F_ST **0.6772**, bit-identical
  to batch 1's `cal_4`. The pipeline is deterministic.
- **15× spread, monotone, and the bolded five form a factor-of-2 ladder in F_ST.** That is the sweep grid.
  Drop long=0.05 — redundant with 0.1.
- **Panmixia is NOT reachable**; F_ST flattens toward ~0.045 (0.05→0.1 halves the mixing but only
  drops F_ST 21%). Offspring are always placed adjacent to a parent, so some structure is intrinsic
  to the model. ~0.017 of that floor is small-block sampling noise, so the true floor is ~0.03.
  0.045 is the "mixed" endpoint; it is good enough, and the range is 15×.
- **Q4 DECISIVE.** Ne_glob across the six arms: 266, 218, 199, 205, 214, 182 — **no relation** to a
  15× change in F_ST. The panmictic estimator is completely blind to fragmentation. The experiment
  needs within-neighbourhood sampling for local Ne; F_ST is the trustworthy structure axis meanwhile.

## THE THREE-ROUTE EXPERIMENT (built 2026-08-15, NOT YET RUN)
`routes_configs.py` + `routes_qsub.sh`. One lattice burn-in per seed at the MOST MIXED setting
(long=0.1, so every arm starts unstructured), then 9 arms branch by resume `--override`:

| arm | override | moves | holds fixed |
|---|---|---|---|
| `ctrl` | — | — | shared baseline for all three |
| `A_ld0200/0050/0010/0000` | `MIGRATION_LONG_RATE` | **Ne** (F_ST 0.09→0.68) | K, N, N·u |
| `B_mu05/20/40` | `G_muta_initpheno` ×0.5/2/4 | **N·u** | K, N, Ne |
| `C_starv` | `REPRODUCTION_REGULATION=false` | **extrinsic mortality** | K, N·u, ~Ne |

9 arms × 3 seeds = 27 phase-2 jobs. K=3000, 100k burn-in + 200k released.
Feasibility facts checked before building: `G_muta_evolvable` defaults to **False** (so the mutation
rate is a fixed parameter and arm B is meaningful), and none of `MIGRATION_LONG_RATE`,
`G_muta_initpheno`, `REPRODUCTION_REGULATION` are in `STRUCTURAL_PARAMETERS`, so all three arms
branch from one ancestor.

⚠️ **`--extend N` is a TOTAL, not an increment** — `init_resume` sets `STEPS_PER_SIMULATION = N`
outright and rejects N ≤ the checkpoint step. Phase 2 must be given `TOTAL=300000` (burn+fwd), not
200000, or every arm is silently truncated to half its released phase. `routes_configs.py` prints
the correct number.

### FUTURE: the REPRODUCTIVE-AGING axis (Ruchitha) — deliberately NOT in this run
Decision 2026-08-15 (Dario): keep `G_repr_evolvable=False` for the runs now in flight, so this
experiment stays on the survival axis. What a reproduction arm will need when it happens:
- ⚠️ **It cannot be added by `--override`.** `G_repr_evolvable` and `G_repr_agespecific` are in
  `STRUCTURAL_PARAMETERS` (the set covers `G_{trait}_{evolvable,agespecific}` for every trait), so
  they change the genome/phenotype array shape. A reproduction arm needs **its own burn-in from
  scratch** — it cannot branch off the current ancestor.
- ⚠️ **Turning repr on needs lo/hi pre-compensation**, exactly as `surv` does. The lo/hi
  double-rescale means the defaults would give an effective range `[0, 0.25]`, not `[0, 0.5]`.
  Use `G_repr_lo: 0, G_repr_hi: 0.7071` (d = sqrt(0.5)). Same trick as `G_surv_lo: 0.4523`.
- Genome grows: `surv` + `repr` + `neut` age-specific loci instead of `surv` + `neut`, so L rises
  and `genetic_ne.py` picks the new L up automatically from `allele_frequencies.csv`.
- The capability is already published: the AEGIS tool paper (Bagic, Šajina, Bradshaw & Valenzano,
  PLOS Comput Biol 2026) reports that somatic **and reproductive** aging evolve spontaneously.
- **Why this axis is the better novelty bet.** Lehtonen 2020 and Aubier & Galipaud 2024 both model
  MORTALITY (lethal, age-specific mutations); Lohr 2014 measured lifespan. None of them treat
  fecundity. Ne → *reproductive* senescence is far less crowded than Ne → lifespan, and Hamilton's
  selection gradients differ between survival and fecundity, so the two need not erode at the same
  rate under drift — which is a testable, genuinely open question.

### ⚠️ A METRIC THAT FAILED — do not repeat
The first calibration analyzer measured isolation-by-distance as *neighbour lineage concordance*
using `lineage_id` from the lattice snapshot. It returned exactly 0 for every arm. **`lineage_id` is
a UNIQUE-PER-INDIVIDUAL pedigree node, not a clan label** — `DEFAULT_PARAMETERS` says so: "each
individual is assigned a unique lineage_id at birth and stores the parent's lineage_id". No two
individuals ever share one, so the statistic was vacuously zero. Founder clans would need
`LINEAGE_RATE > 0` and a walk back through `parent_lineage_id` in `/lineage/births.csv`.
Replaced by block F_ST from the genotype snapshot joined to lattice positions **by row index** —
valid because `latticerecorder` writes `for i in range(n)` over the population arrays and
`featherrecorder` builds from the same population unreordered. The analyzer asserts both the row
counts and the step numbers match rather than trusting it.

### Calibration scripts — `--sweep {migration,longdispersal}`, gate before the fragmentation sweep
Batch 1 (`--sweep migration`, the default) is DONE — results above. Batch 2 maps the
long-dispersal axis, which batch 1 identified as the stronger and wider knob. **Use a separate
CONFIG_DIR per batch**: the qsub globs `cal_*.yml`, so two batches in one directory would interleave
and silently shift the `-t` task numbering. The generator refuses to write into a dir that already
holds `cal_*.yml`.
```
# batch 2 -- long-dispersal axis at fixed MIGRATION_RATE=0.01
CONFIG_DIR=/wins/vlzno/projects/aegis_latcal2
python experiments/ne_lifespan/lattice_calibration_configs.py --outdir $CONFIG_DIR --sweep longdispersal
mkdir -p logs
CONFIG_DIR=$CONFIG_DIR qsub -t 1-6 experiments/ne_lifespan/lattice_calibration_qsub.sh
```
`cal_1_ld0000` repeats batch 1's `cal_4` exactly (m=0.01, long=0) as a free reproducibility check —
it must return F_ST ≈ 0.677. If it does not, the pipeline is not deterministic and nothing else here
can be trusted.
**Pull the genotype snapshots too** — F_ST needs them, and the first pull missed them:
```
rsync -av --prune-empty-dirs --include='*/' --include='cal_*.yml' \
  --include='popsize_after_reproduction.csv' --include='popgen/***' --include='lattice/***' \
  --include='*.feather' --exclude='*' \
  dvalenza@gen100:/wins/vlzno/projects/aegis_latcal2/ ~/aegis_data/latcal2/
~/aegis-venv/bin/python experiments/ne_lifespan/analyze_lattice_calibration.py ~/aegis_data/latcal2/cal_*/
```
6 arms, identical K=3000, 5000 steps, differing only in viscosity. Answers: Q1 does N track K under
lattice regulation; Q2/Q3 does viscosity buy structure range; Q4 does the global Ne estimator go blind.
Isolation-by-distance is measured as neighbour lineage concordance from the lattice snapshot alone
(`q, r, lineage_id` — no join against the genotype feather, so no row-alignment assumption):
`I = (P(same lineage | adjacent) − P(same lineage | random)) / (1 − P(...|random))`, normalised so arms
with different lineage diversity stay comparable. Analyzer is stdlib-only; `--selftest` runs anywhere.

### The three-arm experiment this gates (design agreed with Dario, NOT yet built)
Off one equilibrated lattice ancestor, each arm isolating one path from ecology to life history:
| arm | K | census N | N·u | Ne | isolates |
|---|---|---|---|---|---|
| `MIGRATION_RATE` sweep | fixed | fixed | fixed | **varies** | drift barrier (Route 1) |
| `G_muta_initpheno` sweep at fixed K | fixed | fixed | **varies** | fixed | mutational supply (Route 2) |
| regulation mode (starvation on/off) | fixed | fixed | fixed | ~fixed | extrinsic mortality (Route 3) |

All read out identically: intrinsic e0 from the genetic `surv` phenotype, ideally newborn-conditioned.

## GOTCHAS FOUND WHILE WRITING THE WRAPPER (worth keeping)
- **`POPGENSTATS_RATE` must divide `STEPS_PER_SIMULATION` exactly.** `funcs.skip()` fires only on
  `steps % rate == 0`; unlike snapshots (which `SNAPSHOT_FINAL_COUNT=60` guarantees at the end) popgen
  has NO end-of-run guarantee. A non-dividing rate silently hands the analysis a mid-run diversity
  measurement. `ne_lifespan_configs.popgen_rate_for()` walks the rate down to a divisor.
- **A pickle is always written on the last step** regardless of `PICKLE_RATE`
  (`PickleRecorder.write` special-cases `is_last_step`), so the burn-in ancestor is guaranteed.
- **No `cp -r` per arm needed here**, unlike `oscillation_qsub.sh`. That design had every arm *resume* a
  shared burnt-in directory, and resume appends to the output CSVs in place, so each arm needed its own
  byte copy. Here each arm starts *fresh* from a subsampled pickle (`aegis sim -p`) into its own dir;
  the burn-in is read-only. Much cheaper on the network mount.
- **A job killed before `CHECKPOINT_RATE`** leaves an output dir with no checkpoint, which aegis refuses
  to resume (`FileNotFoundError`). The wrapper detects that and clears the dir instead of failing on
  every resubmission.

**Local env, corrected 2026-08-15:** the *engine* is not installed on the laptop, but `~/aegis-venv`
exists and carries numpy 2.5, pandas 3.0, pyarrow, PyYAML and matplotlib. So **config generation and
ALL analysis run locally** (verified) — only running sims and `subsample.py` need `aegis_sim`:
```bash
~/aegis-venv/bin/python experiments/ne_lifespan/analyze_lattice_calibration.py ...   # works
~/aegis-venv/bin/python experiments/ne_lifespan/ne_lifespan_configs.py --outdir ...  # works
```
`runs/genetic_ne.py --selftest` and `analyze_lattice_calibration.py --selftest` are stdlib-only and
run under the system python too.

## RUNNING ON MERLIN (gen100)

Repo state as of 2026-08-15: `origin` is `https://github.com/valenzano-lab/aegis`;
`exp-oscillation-burnin` is fully pushed and is a strict fast-forward of `v2` (v2 is 0 ahead,
23 behind). Both branches carry `runs/ne_ma_ap_configs.py` and `src/aegis_sim/submodels/lattice.py`,
so either is a valid base — branch off `exp-oscillation-burnin` to also get resume `--override`,
which the three-arm experiment will need.

**Paths below are VERIFIED against records of real runs** (oscillation job 973645 of 2026-07-24;
the Ne×MA/AP sweep of 2026-07-17→20), not assumed:
`~/aegis` = cluster checkout · `dvalenza@gen100` = ssh target · `/wins/vlzno/projects/<name>` =
run storage · `~/aegis_data/<name>/` = where results are pulled to on the Mac (**never** into the
repo — it is inside Dropbox).
**`AEGIS_ENV` VERIFIED 2026-08-15** (it was an open question for a year): the env
`/home/lakatos/dvalenza/.conda/envs/aegis` exists and holds `bin/aegis`, and its `aegis_sim` is an
**editable install pointing at `~/aegis`** — so `git checkout` in that repo updates the engine and
**no `pip install` is needed**. The qsub default is correct. Still `conda activate aegis` before
submitting: the base env has no aegis, and `#$ -V` then carries the right env to the compute nodes
too. (Note `export PATH="${AEGIS_ENV}/bin:${PATH}"` only *prepends* — a wrong AEGIS_ENV would be a
silent no-op that leaves the `-V`-inherited PATH in charge, which is how earlier runs succeeded
without anyone confirming the path.)
⚠️ The `# storage is under /scratch/merlin` comment in `runs/ne_ma_ap_qsub.sh` is **stale**: every
actual run used `/wins/vlzno/projects/`. Corrected 2026-08-15.
SGE array jobs (`#$ -t`) were once flagged untested; the 36-arm oscillation array settled that —
**arrays work.**

```bash
# --- laptop: land the work on a topic branch -------------------------------
git checkout -b exp-ne-lifespan
git add experiments/ runs/genetic_ne.py runs/ne_ma_ap_configs.py
git commit -m "Ne->lifespan: cluster pipeline, genetic-Ne units fix, lattice calibration"
git push -u origin exp-ne-lifespan

# --- gen100 (LOGIN node: submit only, never run a sim here) ----------------
ssh dvalenza@gen100
cd ~/aegis
git fetch origin && git checkout exp-ne-lifespan && git pull
command -v aegis || echo "FIX AEGIS_ENV FIRST -- the default env path is unverified"
pip install -e . --no-deps        # only if the env is NOT already an editable install

# pre-flight: catch a stale engine BEFORE queueing six jobs (cheap, login-node safe)
python -c "from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS as D; \
print([p for p in ('LATTICE_MODE','MIGRATION_RATE','MIGRATION_LONG_RATE', \
'LATTICE_TARGET_DENSITY','LATTICE_RECORD_RATE','LINEAGE_TRACING') if p not in D] or 'engine OK')"

# --- calibration ------------------------------------------------------------
export CONFIG_DIR=/wins/vlzno/projects/aegis_latcal     # OUTSIDE the git tree
python experiments/ne_lifespan/lattice_calibration_configs.py --outdir $CONFIG_DIR
mkdir -p logs
CONFIG_DIR=$CONFIG_DIR qsub -t 1-6 experiments/ne_lifespan/lattice_calibration_qsub.sh
qstat ; tail -f logs/latcal.1.out
python experiments/ne_lifespan/analyze_lattice_calibration.py $CONFIG_DIR/cal_*/
```

**Submit from the repo root.** The scripts use `#$ -cwd`, logs go to `./logs`, and
`ne_lifespan_qsub.sh` phase 2 invokes `python experiments/ne_lifespan/subsample.py` by
RELATIVE path — submitting from elsewhere fails at the subsample step, after the burn-in.

`AEGIS_ENV` defaults to `/home/lakatos/dvalenza/.conda/envs/aegis` in every qsub script; export a
different value before `qsub` for another user or env.

**Disk.** Calibration sets `SNAPSHOT_FINAL_COUNT=1` and is a few MB per run. The MAIN experiment
keeps the default 60 (needed to pool newborns for the newborn-conditioned e0) — budget ~1 GB per
N=10000 forward arm, so roughly 5 GB for the 15-arm sweep.

**Analysis happens on the laptop.** Pull the small series only, into `~/aegis_data/` — **never into
the repo**, which is inside Dropbox:
```bash
rsync -av --prune-empty-dirs \
  --include='*/' --include='output_summary.json' --include='final_config.yml' \
  --include='popsize_after_reproduction.csv' --include='popgen/***' --include='lattice/***' \
  --exclude='*' \
  dvalenza@gen100:/wins/vlzno/projects/aegis_latcal/ ~/aegis_data/latcal/
~/aegis-venv/bin/python experiments/ne_lifespan/analyze_lattice_calibration.py ~/aegis_data/latcal/cal_*/
```
Note the rsync must also bring each run's `<name>.yml` (it sits beside the run dir, so include it or
copy the configs separately) — the analyzer reads `MIGRATION_RATE` from it.

## ENVIRONMENT
Needs an aegis env: python3.11 + `numpy pandas pyyaml pyarrow platformdirs psutil numba`, then
`pip install -e <aegis repo> --no-deps` (GUI deps like dash NOT needed — run the engine via
`aegis_sim.run`, not the `aegis` CLI which imports the GUI). The installed pypi `aegis-sim 2.2` is
too old (pre-`aegis_sim` refactor); use the local repo.
