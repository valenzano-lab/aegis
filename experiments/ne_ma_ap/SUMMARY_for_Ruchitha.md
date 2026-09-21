# Ne × {MA, AP}: results, scripts, and what changes next

For Ruchitha, ahead of the retreat. Everything here is on branch `exp-ne-lifespan`.

## Your role

With Martin we only explored the evolution of **survival**. Your part is **reproduction**.
The plan runs in three stages:

1. Survival only (as in the paper, reproduction fixed; these runs are that setup).
2. Reproduction only (survival fixed).
3. Survival and reproduction co-evolving.

The runs below are all stage 1, so read them as the baseline your reproduction runs will be
compared against.

## Read this first: these results are not the paper's model

In Bagic & Valenzano (bioRxiv 2022), **MA could only accumulate deleterious variants**.
Survival started at 0.95 and every MA effect was negative (`P = G·M + 0.95`, all entries of
M < 0), so MA had a ceiling and only AP could produce beneficial (early-acting) variants.

The model that produced the runs below removed that asymmetry. Survival is encoded directly
(20 bits per age, each bit worth 0.015), bits flip both ways, and survival lives in
[0.7, 1.0] starting from 0.95. **So beneficial variants can also accrue under MA.** You can
see it in the figure: early survival in the MA arm climbs to ~0.997, above the 0.95 start,
which was impossible in the paper.

This is why the MA arm here does not show what the paper showed. For the paper we are
studying the accumulation of *gerovariants* under MA vs AP, so **we are going back to the
paper's construction: MA accumulates deleterious variants only.** The runs below are still
useful (the AP mechanism and the neutral-clock timing both carry over), but read the MA
results as those of a different model.

## What was run

- Composite architecture + phenomap. MA = direct encoding only; AP = the same plus 50
  pleiotropic drivers on the `neut` loci. Each driver raises survival at one age and lowers it at
  another, with 50/50 "Williams" (+early/−late) and anti-Williams (−early/+late) orientation,
  so AP is tested rather than assumed.
- `AGE_LIMIT 50`, `MATURATION_AGE 10`, reproduction fixed at 0.25 (not evolvable), so all
  aging comes from survival.
- Mutation: `G_muta_initpheno 1.7e-4` (~1 functional mutation/genome/generation),
  `MUTATION_RATIO 0.1` (1→0 favoured 10:1).
- `G_surv_lo/hi = 0.4523/1.0`: these values compensate for a known engine quirk (lo/hi
  applied twice) so that the *effective* range is [0.7, 1.0]. See
  `docs/phenotype-lo-hi-double-rescale.md`.
- Sexual, **seed 1 only**, Ne = 300 / 3,000 / 30,000, 1,000,000 steps each. Seeds 2–3 and
  the asexual arm have not been run.

## Results (final snapshot, step 1,000,000)

| Arm | Ne | Life expectancy (stages) | Median lifespan | Survival ages 0–9 | Survival ages 40–49 |
|---|---|---|---|---|---|
| MA | 300 | 26.2 | 29 | 0.984 | 0.759 |
| MA | 3,000 | 39.7 | 44 | 0.996 | 0.833 |
| MA | 30,000 | 42.1 | 47 | 0.997 | 0.922 |
| AP | 300 | 27.4 | 29 | 0.994 | 0.718 |
| AP | 3,000 | 38.8 | 41 | 0.997 | 0.818 |
| AP | 30,000 | 41.5 | 45 | 0.998 | 0.892 |

Life expectancy = Σ lₓ from age 0, from mean per-age survival across individuals.

1. **Larger Ne → longer life in both arms.** Most of the gain is between 300 and 3,000.
2. **AP shows the Williams trade-off.** It beats MA early (ages 0–9) and loses late
   (ages 40–49) at every Ne.
3. **The AP mechanism is confirmed at the locus level.** Selection raised the Williams drivers
   and purged their mirror images (mean dosage, 1 = homozygous):

   | | Ne=300 | Ne=3,000 | Ne=30,000 |
   |---|---|---|---|
   | AP Williams / anti | 0.319 / 0.000 | 0.448 / 0.006 | 0.464 / 0.011 |
   | MA control Williams / anti | 0.064 / 0.053 | 0.074 / 0.069 | 0.124 / 0.073 |

   In MA the same loci are wired to nothing, so they are the drift control. Selection
   efficiency in AP levels off above Ne=3,000: the 64-hour Ne=30,000 run adds little.
4. **Open flag: the MA control at Ne=30,000.** The two driver classes should match, but they
   differ by +0.05 (permutation p = 0.05, n = 25 vs 25 loci). Per-locus values range from 0 to
   0.42 in both classes, which is wide for neutral loci at this Ne. Possible causes are
   hitchhiking with linked survival loci or chance at n = 1. I haven't resolved this yet.
5. **Neutral clock.** The 19 unexpressed bits of each `neut` locus are a clean neutral
   baseline. They relax from 0.5 to the mutation–drift equilibrium of 0.091 in both arms and
   at every Ne, and they get there at **~400,000–500,000 steps**. That is the timing for the
   next design (below).

Figures:
- `survival_mortality_drivers.png`: survival, log-mortality, survivorship, driver dosage.
- `neut_trajectory.png`: neutral load and bit-0 signal over the whole run.

## Proposal for the retreat meeting with Martin

Redo MA as it was in the paper, on a pre-evolved population:

1. **Pre-evolve** one ancestral population until the neutral locus reaches equilibrium, as in
   Bradshaw et al. (bioRxiv). From these runs that is roughly 500,000 steps, but check it with
   `runs/check_equilibration.py` rather than hard-coding it.
2. **Restore the paper's MA:** deleterious variants only, so survival has a ceiling and MA
   cannot accrue beneficial variants.
3. **Branch** the equilibrated population into the MA and AP arms across Ne, so that forward
   differences come from the arm and Ne alone and not from the burn-in.
4. Replicate: at least 3 seeds. In AP, the seed also draws which ages each driver hits.

Design questions to settle at the meeting:
- **How to impose the MA ceiling in the current engine.** One option: start all survival
  bits ON and set `hi` so the all-ON genome equals the start value. Every mutation is then
  deleterious, and a reversion can only restore survival up to the ceiling, never above it.
  This needs no engine change.
- **Does the ceiling apply to AP too?** The paper's AP could exceed 0.95 early; ours shares
  the direct encoding with MA. The arms should still differ only in the pleiotropy.
- **What does the population evolve under during pre-evolution** (MA rules, AP rules, or
  neutral), and does that bias one arm?

## Scripts (branch `exp-ne-lifespan`)

| Script | What it does |
|---|---|
| `runs/ne_ma_ap_configs.py` | Generates all configs (arms × Ne × mode × seed); `ap_specs(seed)` rebuilds each seed's AP driver map |
| `runs/ne_ma_ap_qsub.sh` | SGE array wrapper for Merlin (submit from gen100, never run there) |
| `runs/plot_ne_ma_ap_survival.py` | Life-table + driver-dosage figure; prints the dosage table |
| `runs/plot_neut_trajectory.py` | Neutral load vs bit-0 signal over time |
| `runs/decode_neut_genotypes.py` | Decodes all 20 bits of each `neut` locus from genotype snapshots |
| `runs/check_equilibration.py` | Tests whether a run's neutral load has reached equilibrium |

Reproduce the sweep that produced these results:

```bash
git checkout exp-ne-lifespan
python runs/ne_ma_ap_configs.py --outdir configs/ --steps 1000000 --seeds 1 --modes sexual --ne 300 3000 30000
mkdir -p logs && qsub -t 1-6 runs/ne_ma_ap_qsub.sh
```

Reproduce the figures from the data:

```bash
python runs/plot_ne_ma_ap_survival.py --datadir <data> -o survival_mortality_drivers.png
python runs/plot_neut_trajectory.py   --datadir <data> -o neut_trajectory.png
python runs/decode_neut_genotypes.py  --datadir <data>
```

**Data** (six runs, ~2.1 GB): on the cluster at `/wins/vlzno/projects/aegis_runs/`.

**Gotchas:**
- The seed-1 data predates a recording fix, so its `neut_*` phenotype columns are **half**
  the true dosage. The scripts already correct for this; multiply by 2 if you read them
  yourself.
- MA takes time: its Ne effect was invisible at 10k steps and only showed by 50k. Don't
  judge MA on a short run.
- Runtimes on Merlin: Ne=300 ~40 min; Ne=3,000 ~3–5 h; Ne=30,000 ~64 h.
