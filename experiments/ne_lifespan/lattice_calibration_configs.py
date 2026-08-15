"""Calibration for the fragmentation (lattice) arm of the Ne -> lifespan experiment.

WHY THIS RUNS BEFORE THE EXPERIMENT
    The fragmentation design rests on a claim that has not been tested in this engine:
    that lowering MIGRATION_RATE lowers the drift-relevant effective size while leaving
    carrying capacity, census N and total N*u untouched. If true, it delivers the design
    Dario asked for -- equal K, wide Ne range -- which is impossible in a well-mixed
    AEGIS population (binomial reproduction cannot be overdispersed, so Ne is pinned to
    within ~2x of N). If false, or if lattice mode distorts the demography, the whole
    arm is worthless and we should know that after ~6 cheap runs rather than after a
    full sweep.

WHAT IT MEASURES
    Q1  Does census N track K under LATTICE_MODE, or does local placement failure hold
        it below K / destabilise it? Lattice mode places each offspring in a random
        ADJACENT EMPTY cell and fails the birth if none is free -- a local density
        regulation that operates on top of REPRODUCTION_REGULATION. That is a different
        regulation mechanism from the one the rest of the experiment relies on.
    Q2  Does viscosity actually generate isolation by distance, and how much? Measured
        as neighbour lineage concordance (see analyze_lattice_calibration.py).
    Q3  How far apart do the migration arms sit -- i.e. how much range is there to sweep?
    Q4  Do global, panmixia-assuming Ne estimates diverge from the structure? Under
        fragmentation global theta_w can stay HIGH (Wahlund: variants preserved across
        demes) while local selection efficiency collapses. If runs/genetic_ne.py reports
        an unchanged Ne across the migration arms, that divergence is the finding, not a
        null result.

THE SILENT FAILURE THIS GUARDS AGAINST
    Population.initialize() is the ONLY caller of lattice.assign_initial_positions(), and
    every lattice code path in the bioreactor is guarded by `positions is not None`. So a
    lattice run seeded from a NON-lattice pickle runs perfectly happily as a well-mixed
    population while its config says LATTICE_MODE=True. These calibration runs start
    fresh (no pickle) so positions are always assigned; the analyzer checks that a
    lattice/ directory actually appeared, which is the observable proof it took effect.

Usage:
    python experiments/ne_lifespan/lattice_calibration_configs.py --outdir DIR
"""
import argparse
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "runs"))
from ne_ma_ap_configs import build  # noqa: E402

from ne_lifespan_configs import ARCH, popgen_rate_for  # noqa: E402

# (name, LATTICE_MODE, MIGRATION_RATE, MIGRATION_LONG_RATE)
# Numbered prefixes so `ls` order == the intended qsub array order.
SWEEPS = {
    # BATCH 1 (ran 2026-08-15, job 974216). Local-viscosity axis. Values are the
    # parameter file's own evalrange for MIGRATION_RATE ([0.0, 0.01, 0.1, 0.5]).
    # RESULT: F_ST 0.245 / 0.445 / 0.677 / 0.680 / 0.172 -- monotone, 4x spread, and
    # SATURATED below m=0.01 (m=0 adds nothing). See HANDOFF.md.
    "migration": [
        ("cal_1_wellmixed",     False, None,  None),   # control: non-lattice engine behaviour
        ("cal_2_mig050_ld000",  True,  0.5,   0.0),    # weakest viscosity
        ("cal_3_mig010_ld000",  True,  0.1,   0.0),    # engine default
        ("cal_4_mig001_ld000",  True,  0.01,  0.0),    # strong isolation by distance
        ("cal_5_mig000_ld000",  True,  0.0,   0.0),    # no local migration at all
        ("cal_6_mig001_ld005",  True,  0.01,  0.005),  # strong IBD + rare long dispersal
    ],
    # BATCH 2. Long-dispersal axis at fixed MIGRATION_RATE=0.01.
    # WHY THIS IS THE REAL AXIS: batch 1 showed that adding long=0.005 on top of m=0.01
    # drove F_ST to 0.172 -- BELOW m=0.5 with no long dispersal (0.245). A rare global
    # channel erases more structure than 50x more local diffusion, because local
    # migration is diffusive and mixes slowly across the lattice while long dispersal is
    # global. It also spans further: the local axis saturates, this one should reach
    # panmixia. Batch 1 gave no well-mixed lattice endpoint (island-model Nm < 1 in every
    # arm), so the top values here exist to supply one.
    # cal_1 repeats batch 1's cal_4 exactly (m=0.01, long=0) -- a free reproducibility
    # check: it must come back at F_ST ~= 0.677.
    "longdispersal": [
        ("cal_1_ld0000", True, 0.01, 0.0),
        ("cal_2_ld0010", True, 0.01, 0.001),
        ("cal_3_ld0050", True, 0.01, 0.005),
        ("cal_4_ld0200", True, 0.01, 0.02),
        ("cal_5_ld0500", True, 0.01, 0.05),
        ("cal_6_ld1000", True, 0.01, 0.1),    # aiming past panmixia so the range is bracketed
    ],
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", required=True)
    p.add_argument("--k", type=int, default=3000,
                   help="carrying capacity; identical across arms -- that is the point")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--density", type=float, default=0.3,
                   help="LATTICE_TARGET_DENSITY; 0.3 is the documented baseline")
    p.add_argument("--sweep", choices=sorted(SWEEPS), default="migration",
                   help="which axis to calibrate. Use a SEPARATE --outdir per sweep: the "
                        "qsub globs cal_*.yml, so two batches in one directory would "
                        "interleave and the -t task numbering would silently shift.")
    args = p.parse_args()

    arms = SWEEPS[args.sweep]
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(outdir.glob("cal_*.yml"))
    if existing:
        p.error(f"{outdir} already holds {len(existing)} cal_*.yml configs "
                f"(e.g. {existing[0].name}). Use a fresh --outdir for this sweep, or "
                "delete the old ones -- mixing batches breaks the qsub task numbering.")
    rate = popgen_rate_for(args.steps, n_records=10)

    for name, lattice_mode, mig, mig_long in arms:
        cfg = build(arm="MA", ne=args.k, mode="asexual", seed=args.seed, steps=args.steps)
        cfg.update(ARCH)
        cfg.update(
            INITIAL_POPULATION_SIZE=args.k,
            RESOURCE_MAXIMUM_AMOUNT=args.k,
            RESOURCE_ADDITIVE_GROWTH=args.k,
            STEPS_PER_SIMULATION=args.steps,
            POPGENSTATS_RATE=rate,
            SNAPSHOT_RATE=args.steps,
            PICKLE_RATE=args.steps,
            CHECKPOINT_RATE=args.steps,
            # SNAPSHOT_FINAL_COUNT defaults to 60, i.e. genotype+phenotype+demography
            # feathers for each of the last 60 steps -- ~300 MB per run at K=3000, and
            # the calibration analysis reads none of them (it uses popsize, lattice/ and
            # popgen/). SNAPSHOT_RATE still lands one snapshot on the final step.
            # The MAIN experiment keeps the default 60: pooling newborns across those
            # steps is what makes the newborn-conditioned e0 estimator possible.
            SNAPSHOT_FINAL_COUNT=1,
            # Lineage IDs are what make isolation by distance measurable from the lattice
            # snapshot ALONE -- no join against the genotype feather, so no row-alignment
            # assumption between two recorders.
            LINEAGE_TRACING=True,
            LATTICE_MODE=lattice_mode,
        )
        if lattice_mode:
            cfg.update(
                LATTICE_TARGET_DENSITY=args.density,
                MIGRATION_RATE=mig,
                MIGRATION_LONG_RATE=mig_long,
                # Snapshot the lattice a handful of times; the analyzer uses the last.
                LATTICE_RECORD_RATE=max(1, args.steps // 5),
            )
        yaml.safe_dump(cfg, open(outdir / f"{name}.yml", "w"), sort_keys=True)

    print(f"wrote {len(arms)} calibration configs ({args.sweep} sweep) to {outdir}")
    print(f"  K={args.k} (IDENTICAL across arms), steps={args.steps}, "
          f"seed={args.seed}, POPGENSTATS_RATE={rate}")
    for name, lm, mig, ml in arms:
        print(f"  {name:22} lattice={str(lm):5} migration={mig} long={ml}")
    print()
    print("submit:")
    print(f"  mkdir -p logs")
    print(f"  CONFIG_DIR={outdir} qsub -t 1-{len(arms)} "
          "experiments/ne_lifespan/lattice_calibration_qsub.sh")
    print("analyse (needs the genotype snapshots -- rsync *.feather too):")
    print(f"  python experiments/ne_lifespan/analyze_lattice_calibration.py {outdir}/cal_*/")


if __name__ == "__main__":
    main()
