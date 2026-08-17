"""The three-route experiment: how does ecology reach life history?

THE ARGUMENT. Ruchitha asked whether the aging pattern is driven by carrying capacity K or
by Ne. Dario's position is that population bottlenecks weaken selection, and that this acts
through Ne. Both are right about part of it, because K reaches life history by three
distinct routes and the existing sweep confounds all three:

  Route 1  K -> N -> Ne -> drift barrier         weakens SELECTABILITY (|s| vs 1/2Ne)
  Route 2  K -> N*u      -> mutational supply    changes RAW MATERIAL
  Route 3  K -> resource-limited mortality       steepens the FORCE-OF-SELECTION decline
                                                 (Williams/Medawar; works at infinite Ne)

Each arm here moves exactly one of them off a SHARED equilibrated ancestor:

  arm A (drift)     MIGRATION_LONG_RATE sweep. K, census N and N*u all IDENTICAL; only the
                    spatial structure -- and hence the drift-relevant local size -- varies.
                    This is the arm that was thought impossible: in a well-mixed AEGIS
                    population Ne is pinned within ~2x of N because binomial reproduction
                    cannot be overdispersed. Spatial structure breaks that, because drift
                    is then set by neighbourhood size, not global census.
  arm B (supply)    G_muta_initpheno sweep at fixed K. Moves N*u, leaves Ne untouched.
                    This is Ruchitha's "Case 2", correctly relabelled: varying u does NOT
                    vary Ne. If lifespan responds here, the effect is supply, not drift.
  arm C (mortality) REPRODUCTION_REGULATION=False, so the population overshoots resources
                    and takes proportional survival/reproduction penalties -- the density
                    feedback that opens the extrinsic-mortality channel.

GRID SET BY MEASUREMENT, NOT GUESSWORK. The MIGRATION_LONG_RATE values come from the
lattice calibration (batch 2, 2026-08-15), which mapped them onto F_ST:
    long 0     -> F_ST 0.677      long 0.02  -> F_ST 0.092
    long 0.001 -> F_ST 0.334      long 0.05  -> F_ST 0.058   (dropped: redundant with 0.1)
    long 0.005 -> F_ST 0.172      long 0.1   -> F_ST 0.045
The five kept values form a clean factor-of-2 ladder in F_ST over a 15x range.

WHY THE BURN-IN IS AT THE **MOST MIXED** SETTING. Every arm must start from an ancestor
with no pre-existing spatial structure, so that structure develops FORWARD and between-arm
differences cannot be burn-in history. long=0.1 is the most mixed setting reachable
(F_ST ~ 0.045; the floor is not 0 because offspring are always placed adjacent to a
parent, so some structure is intrinsic to the model).

THE BURN-IN MUST ITSELF BE LATTICE. lattice.assign_initial_positions() is called ONLY from
Population.initialize(), and every lattice path in the bioreactor is guarded by
`positions is not None` -- so a lattice arm branched from a NON-lattice ancestor would run
well-mixed while its config claimed otherwise, silently. See HANDOFF.md.

Usage:
    python experiments/ne_lifespan/routes_configs.py --outdir DIR
"""
import argparse
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "runs"))
from ne_ma_ap_configs import build  # noqa: E402

from ne_lifespan_configs import ARCH, popgen_rate_for  # noqa: E402

MU0 = 1.7e-4          # the pilot mutation rate; arm B scales this
MIXED_LONG = 0.1      # most-mixed reachable setting -> the burn-in regime
MIGRATION = 0.01      # local viscosity, held fixed; the calibration showed it saturates

# name -> resume overrides. "ctrl" is the shared baseline for all three arms: it simply
# continues the burn-in regime, so arm A, B and C each compare against the same control
# rather than three separate ones.
ARMS = {
    "ctrl":     {},
    # arm A -- drift/selectability. F_ST from the calibration in the comment.
    "A_ld0200": {"MIGRATION_LONG_RATE": 0.02},    # F_ST ~ 0.09
    "A_ld0050": {"MIGRATION_LONG_RATE": 0.005},   # F_ST ~ 0.17
    "A_ld0010": {"MIGRATION_LONG_RATE": 0.001},   # F_ST ~ 0.33
    "A_ld0000": {"MIGRATION_LONG_RATE": 0.0},     # F_ST ~ 0.68
    # arm B -- mutational supply. Ne untouched; N*u scaled directly.
    "B_mu05":   {"G_muta_initpheno": MU0 * 0.5},
    "B_mu20":   {"G_muta_initpheno": MU0 * 2.0},
    "B_mu40":   {"G_muta_initpheno": MU0 * 4.0},
    # arm C -- extrinsic mortality. Population overshoots resources and pays proportional
    # surv/repr penalties (the intended density feedback), instead of being birth-capped.
    # ⚠️ C_starv does NOT test extrinsic mortality. ARCH sets STARVATION_PENALTY=0.0 and
    # the multiplier is (1-penalty)**steps = 1.0 always, so dropping the birth cap left the
    # population unregulated: it grew to 9885 (3.3x K) with no mortality penalty at all.
    # Kept as a high-N arm (a real route-1 data point) -- the corrected route-3 arm is below.
    "C_starv":     {"REPRODUCTION_REGULATION": False},
    "C_starv_pen": {"REPRODUCTION_REGULATION": False, "STARVATION_PENALTY": 0.1},
}


def make_burnin_cfg(K, steps, seed, density):
    cfg = build(arm="MA", ne=K, mode="asexual", seed=seed, steps=steps)
    cfg.update(ARCH)
    cfg.update(
        INITIAL_POPULATION_SIZE=K,
        RESOURCE_MAXIMUM_AMOUNT=K,
        RESOURCE_ADDITIVE_GROWTH=K,
        STEPS_PER_SIMULATION=steps,
        POPGENSTATS_RATE=popgen_rate_for(steps),
        SNAPSHOT_RATE=steps,
        PICKLE_RATE=steps,
        CHECKPOINT_RATE=10_000,
        LATTICE_MODE=True,
        LATTICE_TARGET_DENSITY=density,
        MIGRATION_RATE=MIGRATION,
        MIGRATION_LONG_RATE=MIXED_LONG,
        LATTICE_RECORD_RATE=steps // 10,
    )
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--k", type=int, default=3000)
    p.add_argument("--burn-steps", type=int, default=100_000,
                   help="CHECK equilibration afterwards with runs/check_equilibration.py; "
                        "generation time is ~15-20 steps here, so this is ~5-7k generations")
    p.add_argument("--fwd-steps", type=int, default=200_000,
                   help="released steps AFTER the burn-in; total = burn + fwd")
    p.add_argument("--density", type=float, default=0.3)
    args = p.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(outdir.glob("burn_*.yml"))
    if existing:
        p.error(f"{outdir} already holds {len(existing)} burn_*.yml. Use a fresh --outdir.")

    for seed in args.seeds:
        cfg = make_burnin_cfg(args.k, args.burn_steps, seed, args.density)
        yaml.safe_dump(cfg, open(outdir / f"burn_s{seed}.yml", "w"), sort_keys=True)

    n_arms, n_seeds = len(ARMS), len(args.seeds)
    print(f"wrote {n_seeds} burn-in configs to {outdir}")
    print(f"  K={args.k} (IDENTICAL everywhere), burn={args.burn_steps}, fwd={args.fwd_steps}")
    print(f"  lattice ON, MIGRATION_RATE={MIGRATION}, MIGRATION_LONG_RATE={MIXED_LONG} "
          "(most mixed -> arms start unstructured)")
    print(f"\nphase 2 arms ({n_arms}), applied as resume --override:")
    for name, ov in ARMS.items():
        print(f"  {name:10} {ov or '(continues the burn-in regime unchanged)'}")
    print(f"\nsubmit:")
    print(f"  mkdir -p logs")
    print(f"  CONFIG_DIR={outdir} PHASE=1 qsub -t 1-{n_seeds} "
          "experiments/ne_lifespan/routes_qsub.sh")
    print(f"  # CHECK equilibration, THEN:")
    total = args.burn_steps + args.fwd_steps
    print(f"  CONFIG_DIR={outdir} TOTAL={total} PHASE=2 qsub -t 1-{n_arms * n_seeds} "
          "experiments/ne_lifespan/routes_qsub.sh")


if __name__ == "__main__":
    main()
