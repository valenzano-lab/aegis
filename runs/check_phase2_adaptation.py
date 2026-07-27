"""Did the genotype evolve during the released phase, or is the oscillation pure ecology?

Phase 2 releases a population that equilibrated under CONSTANT, regulated resources into
a novel resource regime, and measures 60,000 steps. Those steps contain both the
ecological settling into a limit cycle AND any evolution toward the new regime. If the
selected phenotypes (surv, repr) barely move from release (step 200k) to the end (260k),
the dynamics are ecologically dominated and the scan results stand as ecological facts.
If they move substantially, the released phase is an adaptation transient and a
coevolved design (burn in UNDER each regime, to a phenotype plateau) is needed.

The neutral locus cannot answer this: p* = MUTATION_RATIO/(1+MUTATION_RATIO) depends only
on the mutation ratio, so it is already at equilibrium and stays there regardless of
regime. Adaptation to a regime shows up in the SELECTED traits, so those are what we test.

Reports, per arm, the mean shift in surv and repr across the released phase, and the
change in life expectancy (sum of lx). A shift small relative to the burn-in evolution
(surv went 0.95 flat -> ~0.98 early / ~0.72 late over 200k) means little adaptation.

Usage:
    python runs/check_phase2_adaptation.py --datadir ~/aegis_data/oscillation_scan
"""

import argparse
import pathlib
import re

import numpy as np
import pandas as pd
import yaml

NAME_RE = re.compile(r"burnin_R\d+_s(?P<seed>\d+)_k(?P<k>[\d.]+)_cap(?P<cap>\d+)$")


def schedule(run_dir, step, AL):
    f = run_dir / "snapshots" / "phenotypes" / f"{step}.feather"
    if not f.exists():
        return None
    df = pd.read_feather(f)
    if df.empty:
        return None
    surv = df[[f"surv_{a}" for a in range(AL)]].values.mean(axis=0)
    repr_ = df[[f"repr_{a}" for a in range(AL)]].values.mean(axis=0)
    return surv, repr_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path("~/aegis_data/oscillation_scan").expanduser())
    p.add_argument("--release", type=int, default=200000)
    p.add_argument("--end", type=int, default=260000)
    args = p.parse_args()

    rows = []
    for d in sorted(x for x in args.datadir.iterdir() if x.is_dir()):
        m = NAME_RE.match(d.name)
        if not m:
            continue
        cfg = yaml.safe_load(open(d / "final_config.yml"))
        AL = int(cfg["AGE_LIMIT"])
        mat = int(cfg["MATURATION_AGE"])
        a = schedule(d, args.release, AL)
        b = schedule(d, args.end, AL)
        if a is None or b is None:
            print(f"  skip {d.name:<38} missing snapshot (pull 200000/260000.feather)")
            continue
        (s0, r0), (s1, r1) = a, b
        # early = pre-maturity survival, the trait under strongest selection
        d_surv_early = s1[:mat].mean() - s0[:mat].mean()
        d_surv_late = s1[mat:].mean() - s0[mat:].mean()
        d_repr = r1[mat:].mean() - r0[mat:].mean()
        d_life = s1.cumprod().sum() - s0.cumprod().sum()
        rows.append(dict(seed=int(m["seed"]), k=float(m["k"]), cap=int(m["cap"]),
                         d_surv_early=d_surv_early, d_surv_late=d_surv_late,
                         d_repr=d_repr, d_life=d_life))
        print(f"  {d.name:<38} Δsurv_early {d_surv_early:+.4f}  Δsurv_late {d_surv_late:+.4f}  "
              f"Δrepr {d_repr:+.4f}  Δlife_exp {d_life:+.2f}")

    if not rows:
        raise SystemExit("no arms with both snapshots")
    D = pd.DataFrame(rows)

    print("\n=== magnitude of adaptation over the 60k released phase ===")
    print(f"  |Δsurv_early| mean {D.d_surv_early.abs().mean():.4f}  max {D.d_surv_early.abs().max():.4f}")
    print(f"  |Δsurv_late|  mean {D.d_surv_late.abs().mean():.4f}  max {D.d_surv_late.abs().max():.4f}")
    print(f"  |Δlife_exp|   mean {D.d_life.abs().mean():.2f}  max {D.d_life.abs().max():.2f}")
    print("\n  Reference: burn-in evolved surv from a flat 0.95 to ~0.98 early / ~0.72 late,")
    print("  i.e. late-life surv moved ~0.23 over 200k steps. Compare the |Δ| above to that.")

    print("\n=== is the shift systematic with the regime? (mean Δlife_exp by cap) ===")
    for cap in sorted(D.cap.unique()):
        g = D[D.cap == cap]
        print(f"  cap {cap:>6}: Δlife_exp {g.d_life.mean():+.2f} ± {g.d_life.std():.2f}   "
              f"Δsurv_late {g.d_surv_late.mean():+.4f}")

    biggest = D.d_life.abs().max()
    verdict = ("ECOLOGICAL — genotype barely moved; the scan measures the limit cycle, not adaptation"
               if biggest < 1.0 else
               "ADAPTATION PRESENT — genotype shifted materially; a coevolved design is warranted")
    print(f"\nVERDICT: {verdict}")
    print(f"  (largest single-arm |Δlife_exp| = {biggest:.2f} steps; threshold 1.0)")


if __name__ == "__main__":
    main()
