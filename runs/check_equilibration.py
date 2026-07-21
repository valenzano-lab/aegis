"""Has the population forgotten its initial conditions?

The neutral locus is the objective criterion. It carries no phenotypic effect, so it
relaxes under mutation and drift alone toward

    p* = MUTATION_RATIO / (1 + MUTATION_RATIO)

geometrically per GENERATION (a genome is mutated only at conception):

    p(g) - p* = (p0 - p*) (1 - mu)^g

Note p* does NOT depend on the mutation rate -- mu sets only the relaxation speed,
~ -ln(eps)/mu generations. That independence is exactly why the neutral locus is the
right stopping criterion: it reports how far the system has relaxed without
confounding that with the selective regime.

Reports, per genotype snapshot, the neutral load and the fraction of the initial gap
still remaining, so a burn-in can be stopped on evidence rather than on a guess.

Usage:
    python runs/check_equilibration.py ~/aegis_data/burnin/burnin_const_R2000
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from decode_neut_genotypes import decode, layout_from_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--tol", type=float, default=0.01,
                   help="fraction of the initial gap that may remain (default 0.01 = 99%% relaxed)")
    args = p.parse_args()

    cfg_path = args.run_dir / "final_config.yml"
    if not cfg_path.exists():
        raise SystemExit(
            f"{args.run_dir} has no final_config.yml yet -- the run has not started "
            f"(or the path is wrong). final_config.yml is written when a run begins; "
            f"check `qstat` and try again once it is under way."
        )
    cfg = yaml.safe_load(open(cfg_path))
    ratio = float(cfg["MUTATION_RATIO"])
    p_star = ratio / (1 + ratio)
    p0 = float(cfg["G_neut_initgeno"])
    mu = float(cfg["G_muta_initpheno"])
    layout = layout_from_config(args.run_dir)

    print(f"run           {args.run_dir.name}")
    print(f"MUTATION_RATIO {ratio}  ->  p* = {p_star:.4f}   (start {p0}, gap {abs(p0 - p_star):.4f})")
    print(f"mu = {mu}  ->  {-np.log(args.tol)/mu:,.0f} generations for {(1-args.tol)*100:.0f}% relaxation")
    print(f"layout: bits={layout[0]} n_loci={layout[1]} neut at {layout[2]}..{layout[2]+layout[3]}\n")

    snaps = sorted((args.run_dir / "snapshots" / "genotypes").glob("*.feather"),
                   key=lambda x: int(x.stem))
    if not snaps:
        raise SystemExit("no genotype snapshots yet")

    print(f"{'step':>9}{'neut load':>12}{'gap left':>11}{'status':>14}")
    reached = None
    for s in snaps:
        df = pd.read_feather(s)
        if df.empty:
            continue
        _, load = decode(df, layout=layout)
        obs = load.mean()
        remaining = abs(obs - p_star) / abs(p0 - p_star)
        ok = remaining <= args.tol
        if ok and reached is None:
            reached = int(s.stem)
        print(f"{int(s.stem):>9}{obs:>12.4f}{remaining:>10.1%}{'  EQUILIBRATED' if ok else '  relaxing':>14}")

    print()
    if reached is None:
        print(f"NOT equilibrated yet — keep running (aegis sim -c <cfg> -r --extend N)")
    else:
        print(f"Equilibrated from step {reached:,} — branch experimental arms from a "
              f"checkpoint at or after this step.")


if __name__ == "__main__":
    main()
