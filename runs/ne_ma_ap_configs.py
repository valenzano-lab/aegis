"""Generate configs for the Ne x {MA, AP} x {sexual, asexual} sweep.

This is the model validated on 2026-07-16 (see docs/phenotype-lo-hi-double-rescale.md
for the bug this design avoids). It supersedes the driver/matrix model of
Bagic & Valenzano 2022 (doi:10.1101/2022.12.17.520867) while reproducing its findings.

Design, in short:
  - Survival is encoded DIRECTLY: one locus per age, BITS_PER_LOCUS bits per locus.
    A bit flip is worth (hi - lo) / BITS_PER_LOCUS = 0.015 -- uniform for every bit.
    Bits flip both ways, so survival can fall AND climb: there is no ceiling.
  - The fitness effect of a mutation comes from WHERE it lands, not from a drawn
    effect size. A bit at age 5 is under strong selection; the same bit at age 45
    is nearly neutral. The age structure IS the distribution of fitness effects,
    which is the thing AEGIS exists to isolate -- so nothing is drawn from a DFE.
  - Mutation is blind: a rate (G_muta_initpheno) and a direction bias
    (MUTATION_RATIO). Selection acts only after a bit has flipped.
  - MA and AP differ in exactly one thing: the off-diagonal of the genotype-to-
    phenotype matrix M. MA is M = I. Same architecture, same mutations otherwise.

Parameter provenance (nothing here is tuned to make it work):
  G_muta_initpheno  1.7e-4  ~1 functional mutation/genome/generation, from the human
                            de novo rate (~70 per genome, ~1.5% coding; Kong et al. 2012)
  G_surv_initgeno   0.833   -> survival 0.95 at every age = the "non-aging" start
                            (5% mortality/stage). R0 = 1.57, viable.
  G_surv_lo/hi      0.4523/1.0  pre-compensated so the EFFECTIVE range is [0.7, 1.0];
                            see docs/phenotype-lo-hi-double-rescale.md
  MUTATION_RATIO    0.1     1->0 favoured 10:1
  BITS_PER_LOCUS    20      sets the effect size; the one free resolution knob

Usage:
    python runs/ne_ma_ap_configs.py --outdir configs/
    aegis sim -c configs/AP_Ne3000_sexual_seed1.yml
"""
import argparse
import pathlib

import numpy as np
import yaml

AGE_LIMIT = 50
DRIVER_WEIGHT = 1.0 / 20  # one active driver == one bit == 0.015 of survival

NE_VALUES = [300, 3000, 30000]
ARMS = ["MA", "AP"]
MODES = ["sexual", "asexual"]


def ap_specs(seed):
    """AP drivers: one per age, two effects, 50/50 (+early,-late) / (-early,+late).

    The 50/50 split makes AP a TEST rather than an assumption: selection decides
    which quadrant spreads, instead of us pre-loading only Williams-type genes.
    Ages are drawn t1 < t2 and signs assigned, which reduces to the paper's rule
    (beneficial strictly before detrimental) in the (+,-) quadrant.
    """
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(1, AGE_LIMIT + 1):
        t1, t2 = sorted(rng.choice(np.arange(1, AGE_LIMIT + 1), size=2, replace=False))
        s1, s2 = (+1, -1) if rng.random() < 0.5 else (-1, +1)
        specs.append(["neut", i, "surv", str(int(t1)), str(float(s1 * DRIVER_WEIGHT))])
        specs.append(["neut", i, "surv", str(int(t2)), str(float(s2 * DRIVER_WEIGHT))])
    return specs


def build(arm, ne, mode, seed, steps):
    cfg = dict(
        RANDOM_SEED=seed,
        STEPS_PER_SIMULATION=steps,
        AGE_LIMIT=AGE_LIMIT,
        MATURATION_AGE=10,
        # Ne is set by the resource limit
        INITIAL_POPULATION_SIZE=ne,
        RESOURCE_MAXIMUM_AMOUNT=ne,
        RESOURCE_ADDITIVE_GROWTH=ne,
        # architecture
        GENARCH_TYPE="composite",
        BITS_PER_LOCUS=20,
        # survival: evolvable, age-specific, starts flat at 0.95
        G_surv_evolvable=True,
        G_surv_agespecific=True,
        G_surv_interpreter="binary",
        G_surv_initgeno=0.833,
        G_surv_lo=0.4523,
        G_surv_hi=1.0,
        # reproduction: LOCKED, as in the paper -- all aging is attributable to survival
        G_repr_evolvable=False,
        G_repr_initpheno=0.25,
        # AP drivers (present in both arms so the genome, and hence the mutational
        # load, is identical; in the MA arm they are simply wired to nothing)
        G_neut_evolvable=True,
        G_neut_agespecific=True,
        G_neut_interpreter="single_bit",
        G_neut_initgeno=0.5,
        # mutation: blind. rate + direction bias, nothing else.
        MUTATION_RATIO=0.1,
        G_muta_initpheno=1.7e-4,
        # reproduction mode
        REPRODUCTION_MODE=mode,
        RECOMBINATION_RATE=0.5 if mode == "sexual" else 0,
        # recording. CHECKPOINT_RATE matters: a cluster timeout must not cost the run.
        CHECKPOINT_RATE=100_000,
        PICKLE_RATE=100_000,
        SNAPSHOT_RATE=100_000,
        POPGENSTATS_RATE=0,
        INTERVAL_RATE=100_000,
        LOGGING_RATE=100_000,
    )
    if arm == "AP":
        # M = I + off-diagonal pairs.  MA leaves M = I (no specs at all).
        cfg["PHENOMAP_SPECS"] = ap_specs(seed)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="configs")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES,
                   help="reproduction modes to emit (default: both)")
    p.add_argument("--ne", type=int, nargs="+", default=NE_VALUES,
                   help="population sizes to emit (default: 300 3000 30000)")
    args = p.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for arm in ARMS:
        for ne in args.ne:
            for mode in args.modes:
                for seed in args.seeds:
                    name = f"{arm}_Ne{ne}_{mode}_seed{seed}"
                    with open(out / f"{name}.yml", "w") as f:
                        yaml.safe_dump(build(arm, ne, mode, seed, args.steps), f, sort_keys=True)
                    n += 1
    print(f"wrote {n} configs to {out}/ ({len(ARMS)} arms x {len(args.ne)} Ne "
          f"x {len(args.modes)} modes x {len(args.seeds)} seeds), {args.steps} stages each")
    print("\nsubmit with:  qsub -t 1-%d runs/ne_ma_ap_qsub.sh" % n)
    for i, f in enumerate(sorted(out.glob("*.yml")), start=1):
        print(f"  task {i:>2}  {f.name}")


if __name__ == "__main__":
    main()
