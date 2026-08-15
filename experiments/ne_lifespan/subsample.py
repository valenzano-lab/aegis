"""Draw N individuals at random from a burn-in pickle and save them as a seed population.

This is stage (b) of the cluster pipeline -- the step that creates the COMMON ANCESTOR
relationship the whole experiment rests on. Every forward arm of a given seed draws from
the SAME burn-in pickle, so the arms start from one evolved genetic background and their
later divergence cannot be burn-in history.

The draw is deterministic in (rng_seed, N): a task that is killed and resubmitted
reproduces its own subsample, and the whole experiment is re-runnable from the configs.

Usage:
    subsample.py BURN_PICKLE N OUT_PKL [--rng-seed S]
"""
import argparse
import pathlib

import numpy as np

from aegis_sim.dataclasses.population import Population


def main():
    p = argparse.ArgumentParser()
    p.add_argument("burn_pickle", type=pathlib.Path)
    p.add_argument("n", type=int)
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--rng-seed", type=int, default=0,
                   help="combined with N so each (seed, N) arm draws reproducibly")
    args = p.parse_args()

    if not args.burn_pickle.exists():
        raise SystemExit(f"ABORT: no burn-in pickle at {args.burn_pickle}")

    pop = Population.load_pickle_from(args.burn_pickle)
    n_burn = len(pop)
    if args.n > n_burn:
        raise SystemExit(
            f"ABORT: burn-in ended with {n_burn} individuals, cannot draw {args.n}. "
            "Either the burn-in K was too low or the population crashed -- investigate "
            "before rerunning; do not quietly shrink the target.")

    rng = np.random.default_rng([args.rng_seed, args.n])
    idx = rng.choice(n_burn, size=args.n, replace=False)
    pop *= idx  # in-place subset; goes through phenotypes.keep, so all arrays stay aligned
    assert len(pop) == args.n, (len(pop), args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pop.save_pickle_to(args.out)
    print(f"subsampled {args.n} of {n_burn} from {args.burn_pickle} -> {args.out}")


if __name__ == "__main__":
    main()
