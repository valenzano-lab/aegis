"""Generate cluster configs for the Ne -> lifespan experiment (full range 1e2..1e4).

THE DESIGN (Dario). One burn-in per seed at K_BURN gives every Ne treatment an
IDENTICAL genetic ancestor. Each forward arm subsamples that ancestor to N and runs
at CONSTANT K=N, birth-regulated (no starvation). Forward divergence in evolved
lifespan is therefore attributable to Ne alone -- not to independent trajectories and
not to the extrinsic-mortality channel that density-regulation MODE opens up (see
HANDOFF.md: every starvation regime evolved e0~15-18, every birth-regulated regime
e0~19-20 at matched K, so the regulation mode MUST be held fixed).

This is the cluster-shaped version of preevolve.py, which is a single-process driver
(burn-in -> subsample -> forward, all in one main()) and cannot be mapped onto an SGE
array. Here the three stages are separate artefacts:

    ne_lifespan_configs.py   -> burn_s{seed}.yml, fwd_N{N}_s{seed}.yml   (this file)
    subsample.py             -> sub_N{N}_s{seed}.pkl from the burn-in pickle
    ne_lifespan_qsub.sh      -> PHASE=1 burn-ins, PHASE=2 subsample+forward array

Architecture is inherited from runs/ne_ma_ap_configs.build(arm="MA", mode="asexual"),
then overridden by ARCH below, so the cluster sweep and the validated local pilot are
the same model. MA/asexual: no phenomap, no recombination -- the mutation-load channel
in isolation.

Usage:
    python experiments/ne_lifespan/ne_lifespan_configs.py --outdir /wins/vlzno/projects/aegis_ne_lifespan
    python experiments/ne_lifespan/ne_lifespan_configs.py --outdir DIR --seeds 1 2 3 \
        --targets 100 316 1000 3162 10000 --burn-steps 200000 --fwd-steps 200000
"""
import argparse
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "runs"))
from ne_ma_ap_configs import build  # noqa: E402

# Overrides on top of the ne_ma_ap architecture. These are the pilot's values -- the
# ones the interim Ne->e0 table was measured under. Do not drift from them silently:
# AGE_LIMIT in particular sets the genome shape, so changing it makes the cluster
# results incomparable to the pilot (and multiplies compute).
ARCH = dict(
    AGE_LIMIT=30,
    MATURATION_AGE=6,
    # MAX_OFFSPRING_NUMBER defaults to 1, which makes boom-bust impossible and pins
    # Ne ~= N regardless of regime. >1 is required for any demographic Ne effect.
    MAX_OFFSPRING_NUMBER=3,
    # Birth-regulated, no starvation death: N is held at K, so the extrinsic-mortality
    # channel is shut off and Ne is the only thing varying across arms.
    STARVATION_PENALTY=0.0,
    REPRODUCTION_REGULATION=True,
    # PLOIDY is left at its default of 2, so "asexual" here is CLONAL DIPLOID
    # (RECOMBINATION_RATE=0), not haploid -- and this sample size therefore counts
    # CHROMOSOMES, i.e. 100 chromosomes ~= 50 individuals. At the N=100 arm that is
    # 100 of 200 chromosomes; at N=10000 it is 100 of 20000. The estimator divides by
    # a_n = harmonic(nsample-1) per record, so the arms stay comparable -- only the
    # sampling variance differs. See runs/genetic_ne.py for the full units argument.
    POPGENSTATS_SAMPLE_SIZE=100,
)

# 5 points, ~half a decade apart, spanning two orders of magnitude.
TARGETS = [100, 316, 1000, 3162, 10000]
SEEDS = [1, 2, 3]


def make_cfg(K, steps, seed, popgen_rate):
    cfg = build(arm="MA", ne=K, mode="asexual", seed=seed, steps=steps)
    cfg.update(ARCH)
    cfg.update(
        INITIAL_POPULATION_SIZE=K,
        RESOURCE_MAXIMUM_AMOUNT=K,
        RESOURCE_ADDITIVE_GROWTH=K,
        STEPS_PER_SIMULATION=steps,
        POPGENSTATS_RATE=popgen_rate,
        # A pickle is always written on the last step regardless of rate
        # (PickleRecorder.write special-cases is_last_step), but keep the rate coarse
        # so the burn-in does not litter the run with multi-hundred-MB pickles.
        PICKLE_RATE=steps,
        SNAPSHOT_RATE=steps,
        # Cluster wall-clock insurance: a killed job must resume, not restart.
        CHECKPOINT_RATE=10_000,
    )
    return cfg


def popgen_rate_for(steps, n_records=20):
    """Choose POPGENSTATS_RATE so a record lands EXACTLY on the final step.

    funcs.skip() fires only when steps % RATE == 0 -- unlike snapshots, popgen has no
    end-of-run guarantee (SNAPSHOT_FINAL_COUNT=60 covers snapshots). Analysis reads the
    LAST row of popgen/simple.csv, so a rate that does not divide STEPS_PER_SIMULATION
    silently hands the analysis a mid-run diversity measurement.
    """
    rate = max(1000, steps // n_records)
    while steps % rate:
        rate -= 1  # walk down to the nearest divisor; bounded, steps % 1 == 0 always
    return rate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", required=True,
                   help="where configs (and therefore output) land; keep OUTSIDE the git tree")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--targets", type=int, nargs="+", default=TARGETS,
                   help="forward population sizes N (= K, birth-regulated)")
    p.add_argument("--k-burn", type=int, default=None,
                   help="burn-in carrying capacity (default: max target)")
    p.add_argument("--burn-steps", type=int, default=200_000)
    p.add_argument("--fwd-steps", type=int, default=200_000,
                   help="long enough for mutation load to accumulate at the LOW-Ne end")
    args = p.parse_args()

    k_burn = args.k_burn or max(args.targets)
    if max(args.targets) > k_burn:
        p.error(f"cannot subsample {max(args.targets)} from a burn-in of {k_burn}")
    if max(args.targets) == k_burn:
        print(f"note: the N={k_burn} arm subsamples the WHOLE burn-in population "
              "(a permutation, not a draw) -- it is the ancestor itself, run on. That is "
              "intended: it is the no-drift-reduction end of the range.")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    burn_rate = popgen_rate_for(args.burn_steps)
    fwd_rate = popgen_rate_for(args.fwd_steps)

    written = []
    for seed in args.seeds:
        cfg = make_cfg(K=k_burn, steps=args.burn_steps, seed=seed, popgen_rate=burn_rate)
        path = outdir / f"burn_s{seed}.yml"
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=True)
        written.append(path)

    # N is zero-padded so the qsub array's `ls | sed -n "${TASK_ID}p"` ordering is
    # ascending in N, not lexicographic ("10000" < "316" as a string). That puts the
    # cheap arms at low task IDs, so a partial submission (-t 1-6) is a usable low-Ne
    # subset rather than the three most expensive runs.
    width = len(str(max(args.targets)))
    for N in sorted(args.targets):
        for seed in args.seeds:
            cfg = make_cfg(K=N, steps=args.fwd_steps, seed=seed, popgen_rate=fwd_rate)
            path = outdir / f"fwd_N{N:0{width}d}_s{seed}.yml"
            yaml.safe_dump(cfg, open(path, "w"), sort_keys=True)
            written.append(path)

    n_burn, n_fwd = len(args.seeds), len(args.seeds) * len(args.targets)
    print(f"wrote {len(written)} configs to {outdir}")
    print(f"  burn-in : {n_burn} (K={k_burn}, {args.burn_steps} steps, "
          f"POPGENSTATS_RATE={burn_rate})")
    print(f"  forward : {n_fwd} (N={args.targets}, {args.fwd_steps} steps, "
          f"POPGENSTATS_RATE={fwd_rate})")
    print()
    print("submit (phase 2 ONLY after phase 1 is complete AND checked):")
    print(f"  mkdir -p logs")
    print(f"  CONFIG_DIR={outdir} PHASE=1 qsub -t 1-{n_burn}  experiments/ne_lifespan/ne_lifespan_qsub.sh")
    print(f"  CONFIG_DIR={outdir} PHASE=2 qsub -t 1-{n_fwd} experiments/ne_lifespan/ne_lifespan_qsub.sh")


if __name__ == "__main__":
    main()
