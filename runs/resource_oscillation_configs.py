"""Configs for the variable-resource oscillation experiment.

Tests whether current AEGIS reproduces the population/resource dynamics of
Sajina & Valenzano 2016 (arXiv:1602.00723), Figure 5A: under VARIABLE resources,
population and resources oscillate with large amplitude during an early "burnout"
phase, and -- if the fixed increment Rbar is large -- the oscillations damp as the
genome evolves toward a stable equilibrium. With small Rbar they never damp.

The paper's resource rule (p.2):

    R(t+1) = (R_t - N_t)*k + Rbar     if N_t <  R_t
    R(t+1) = (R_t - N_t)   + Rbar     if N_t >= R_t   (floored at 0)

Current AEGIS implements the first branch exactly. run_step order is
scavenge -> mortalities -> replenish, and replenish is
capacity*(1 + RESOURCE_MULTIPLICATIVE_GROWTH) + RESOURCE_ADDITIVE_GROWTH, so

    k    = 1 + RESOURCE_MULTIPLICATIVE_GROWTH
    Rbar = RESOURCE_ADDITIVE_GROWTH

CONSTANT vs VARIABLE is decided by RESOURCE_MAXIMUM_AMOUNT alone:
  - MAX == ADD  -> pool is pinned at ADD every step (constant resources)
  - MAX is None -> leftover accumulates and is drawn down (variable resources)

Two known deviations from the paper, to keep in mind when reading results:
  1. On overshoot AEGIS zeroes the pool, so R(t+1) = Rbar exactly; the paper
     subtracts the deficit FROM Rbar. AEGIS therefore damps crashes slightly.
  2. Starvation here compounds over consecutive deficit steps
     ((1 - STARVATION_PENALTY) ** n, applied to surv AND repr) rather than the
     paper's constant mu. If anything this should favour oscillation.

Both surv and repr are evolvable: the paper's stabilization IS the genome
evolving, so locking repr (as the Ne x MA/AP model does) would remove the
mechanism under test.

lo/hi are pre-compensated for the phenotype double-rescale (see
docs/phenotype-lo-hi-double-rescale.md): for a true range [L, H] set
d = sqrt(H - L), lo = L/(1+d), hi = lo + d.
  surv true [0.7, 1.0] -> lo 0.4523, hi 1.0
  repr true [0.0, 0.5] -> lo 0.0,    hi 0.7071

Usage:
    python runs/resource_oscillation_configs.py --outdir ~/aegis_data/resource_oscillation
"""

import argparse
import math
import pathlib

import yaml

AGE_LIMIT = 50
MATURATION_AGE = 10
STEPS = 60_000  # the paper's run length; stabilization is not visible earlier


def repr_hi(true_hi):
    """Invert the double-rescale so the effective repr range is [0, true_hi]."""
    d = math.sqrt(true_hi)
    return round(d, 4)  # lo = 0 => hi = d


def build(rbar, variable, mult, steps, age_limit=AGE_LIMIT, maturation=MATURATION_AGE):
    cfg = dict(
        RANDOM_SEED=1,
        STEPS_PER_SIMULATION=steps,
        AGE_LIMIT=age_limit,
        MATURATION_AGE=maturation,
        INITIAL_POPULATION_SIZE=int(rbar),
        # --- the experiment ---
        RESOURCE_ADDITIVE_GROWTH=float(rbar),          # Rbar
        RESOURCE_MULTIPLICATIVE_GROWTH=float(mult),    # k - 1
        RESOURCE_INITIAL_AMOUNT=float(rbar),
        # inf, not None: resources.py accepts None but the parameter validator
        # requires a float. inf is a valid float and yaml round-trips it as .inf,
        # so the cap simply never binds -- the paper's uncapped pool.
        RESOURCE_MAXIMUM_AMOUNT=float("inf") if variable else float(rbar),
        # THE mechanism that makes crashes deep. Without it a depleted pool returns
        # to the full increment in one step, so shortages last 1-2 steps, the
        # compounding starvation penalty never compounds, and N dips only ~45%.
        # With it: shortages run ~6 steps and N crashes ~96% (measured, 4k steps).
        # Off for the constant arm: there the paper sets resources the same at every
        # stage 'regardless from the leftover resources from the previous stage'.
        RESOURCE_DEFICIT_CARRYOVER=bool(variable),
        # Overshoot MUST be allowed: True hard-caps the population at the resource
        # level, so N can never exceed R and the whole feedback disappears.
        REPRODUCTION_REGULATION=False,
        STARVATION_PENALTY=0.1,
        # --- genetics: both surv and repr evolve ---
        GENARCH_TYPE="composite",
        BITS_PER_LOCUS=20,
        G_surv_evolvable=True,
        G_surv_agespecific=True,
        G_surv_interpreter="binary",
        G_surv_initgeno=0.833,   # -> surv 0.95 at every age; R0 ~ 2.6, comfortably viable
        G_surv_lo=0.4523,
        G_surv_hi=1.0,
        G_repr_evolvable=True,
        G_repr_agespecific=True,
        G_repr_interpreter="binary",
        G_repr_initgeno=0.5,     # -> repr 0.25
        G_repr_lo=0.0,
        G_repr_hi=repr_hi(0.5),
        # Neutral loci: no phenotypic effect, so they evolve under mutation + drift
        # alone. This is the ruler that distinguishes the two explanations for the
        # low early-life survival in Fig 5B of the paper:
        #   H1 (buffering)  early surv falls, neutral load UNCHANGED -> selective
        #   H2 (drift/MA)   early surv falls AND neutral load rises  -> weak purifying
        #                   selection at depressed Ne, i.e. the paper's own reading
        # Decode with runs/decode_neut_genotypes.py (note: it assumes surv+neut only,
        # so its N_SURV/N_LOCI constants need adjusting for this 3-trait layout).
        G_neut_evolvable=True,
        G_neut_agespecific=True,
        G_neut_interpreter="single_bit",
        G_neut_initgeno=0.5,
        # Paper's mutation rate (0.001/site). The Ne x MA/AP model uses 1.7e-4,
        # which is likely too slow for the genome to reach equilibrium in 60k stages
        # -- and the evolutionary stabilization is exactly what we are testing for.
        G_muta_evolvable=False,
        G_muta_initpheno=0.001,
        MUTATION_RATIO=0.1,
        REPRODUCTION_MODE="sexual",
        RECOMBINATION_RATE=0.5,
        # --- recording: N(t) and R(t) are written every step by the popsize and
        # resource recorders (no rate parameter), which is all this experiment needs.
        # Everything heavy is turned down to keep the output small.
        SNAPSHOT_RATE=10_000,
        SNAPSHOT_FINAL_COUNT=0,
        PICKLE_RATE=steps,
        POPGENSTATS_RATE=0,
        INTERVAL_RATE=1_000,
        TE_RATE=10_000,
        LOGGING_RATE=5_000,
        CHECKPOINT_RATE=0,
    )
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="~/aegis_data/resource_oscillation")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--rbar", type=int, nargs="+",
                   help="Rbar values for a Ne-descent sweep (variable + constant, all seeds)")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--age-limit", type=int, default=AGE_LIMIT,
                   help="max age. Paper geometry: 70 (ours: 50)")
    p.add_argument("--maturation", type=int, default=MATURATION_AGE,
                   help="age at first reproduction. Paper geometry: 16 (ours: 10)")
    args = p.parse_args()

    out = pathlib.Path(args.outdir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # NOTE on k (= 1 + mult): k > 1 is not usable in an uncapped pool. When R > N the
    # pool compounds at k per step, which no age-structured population with maturation
    # age 10 can consume: a k=1.1 test reached R = 3.7e11 by step 200. The paper's k
    # must therefore be <= 1. k = 1 gives resource peaks ~2.4x Rbar against the
    # paper's ~4x, so it is the right regime; scan DOWNWARD (k=0.5) if needed.
    if args.rbar:
        # Ne-descent sweep: does low Ne under VARIABLE resources depress early-life
        # survival, as in Fig 5B? Constant controls at the same Rbar separate the
        # effect of low Ne from the effect of oscillation itself. Note that an
        # oscillating population's drift is governed by the HARMONIC mean of N,
        # which sits well below its arithmetic mean -- so a variable run has a lower
        # effective Ne than a constant run at the same Rbar.
        runs = []
        for rbar in args.rbar:
            for variable in (True, False):
                for seed in args.seeds:
                    kind = "var" if variable else "const"
                    runs.append((f"{kind}_R{rbar}_s{seed}", rbar, variable, 0.0, seed))
    else:
        runs = [
            ("var_large",   5000, True,  0.0, 1),
            ("var_small",   1000, True,  0.0, 1),
            ("const_large", 5000, False, 0.0, 1),
            ("const_small", 1000, False, 0.0, 1),
        ]

    for name, rbar, variable, mult, seed in runs:
        cfg = build(rbar, variable, mult, args.steps, args.age_limit, args.maturation)
        cfg["RANDOM_SEED"] = seed
        with open(out / f"{name}.yml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        kind = "variable" if variable else "constant"
        print(f"  {name:<18} {kind:<8} Rbar={rbar:<5} seed={seed}")

    print(f"\nwrote {len(runs)} configs to {out}/  ({args.steps:,} stages each)")
    print("run:  for c in %s/*.yml; do aegis sim -c $c; done" % out)


if __name__ == "__main__":
    main()
