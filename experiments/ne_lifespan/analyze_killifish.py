"""Did the water window put a knee in the evolved survival curve, and does it sit at W?

THE PREDICTION. A pool that dries every W steps makes every age >= W invisible to selection.
Deleterious mutations affecting survival there should accumulate, so the evolved per-age
survival px should be HELD below W and ERODED at and above it -- and the boundary should
move with W across arms.

WHY NOT A BREAKPOINT FIT. Fitting a bent line to px(age) is fragile: the earlier peek at the
fragmentation arms showed per-age scatter at late ages an order of magnitude larger than at
early ages (weak selection there means the loci drift freely), and a breakpoint estimator
will happily place a knee in noise. Instead this compares each arm to a MATCHED REFERENCE
arm age by age:

    dpx(age) = px_arm(age) - px_reference(age)

The prediction is then dpx ~ 0 below W and dpx < 0 at/above W -- a sign change at a known
location, not a free parameter. If the signal is weak you get a noisy flat line rather than a
spurious knee.

CHOICE OF REFERENCE -- this matters. K_ctrl has no window AND continuous generations, so
comparing K_W12_annual against it confounds the horizon with the change in generation
structure. The clean within-mode reference is the W=30 arm: same synchrony, same egg-bank
behaviour, but its window coincides with AGE_LIMIT=30 and so imposes no horizon inside the
age range. Both comparisons are reported; the W30 one is the one to believe.

THE CONFOUND TO KEEP IN VIEW. A synchronous cohort is immature for MATURATION_AGE steps, so
a short window leaves fewer reproductive steps per season, a smaller egg bank, and a smaller
standing population. Short W therefore lowers the horizon AND Ne together. The knee POSITION
is diagnostic of the horizon (Ne cannot place a knee at W); the knee DEPTH is confounded.
Realized N is reported per arm so the size of that confound is visible rather than implicit.

Usage:
    python experiments/ne_lifespan/analyze_killifish.py --datadir /wins/vlzno/projects/aegis_routes
"""
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

WINDOWS = [12, 18, 24, 30]
MODES = ["annual", "overlap"]
MATURATION_AGE = 6


def px_of(run_dir):
    """Mean per-age genetic survival from the LAST phenotype snapshot.

    Read off the genetic phenotype, never realized deaths: realized mortality here is
    dominated by the water window itself, which would trivially reproduce the treatment.
    """
    d = pathlib.Path(run_dir) / "snapshots" / "phenotypes"
    if not d.is_dir():
        return None
    snaps = sorted(d.glob("*.feather"), key=lambda p: int(p.stem))
    if not snaps:
        return None
    p = pd.read_feather(snaps[-1])
    cols = sorted([c for c in p.columns if c.startswith("surv_")],
                  key=lambda c: int(c.split("_")[1]))
    return p[cols].mean(axis=0).values


def realized_n(run_dir, tail=2000):
    f = pathlib.Path(run_dir) / "popsize_after_reproduction.csv"
    if not f.exists():
        return None
    ns = pd.read_csv(f, header=None)[0].values.astype(float)[-tail:]
    return float(ns.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="/wins/vlzno/projects/aegis_routes")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    base = pathlib.Path(args.datadir)

    def arm(seed, name):
        return base / f"burn_s{seed}_K_{name}"

    # ---- per-arm summary ---------------------------------------------------
    print("REALIZED POPULATION AND LIFESPAN PER ARM")
    print(f"{'arm':<16}{'seed':>5}{'N':>8}{'e0':>8}")
    for mode in MODES:
        for W in WINDOWS:
            for s in args.seeds:
                d = arm(s, f"W{W}_{mode}")
                px = px_of(d)
                if px is None:
                    continue
                print(f"{'W%d_%s' % (W, mode):<16}{s:>5}{realized_n(d) or float('nan'):>8.0f}"
                      f"{np.cumprod(px).sum():>8.2f}")
    for s in args.seeds:
        d = arm(s, "ctrl")
        px = px_of(d)
        if px is not None:
            print(f"{'ctrl':<16}{s:>5}{realized_n(d) or float('nan'):>8.0f}"
                  f"{np.cumprod(px).sum():>8.2f}")

    # ---- the knee test -----------------------------------------------------
    # dpx against the within-mode W=30 arm (no horizon inside the age range) and, for
    # completeness, against K_ctrl. Below/at-or-above split is at W, the PREDICTED location.
    for ref_kind in ("W30 same-mode", "ctrl"):
        print(f"\nKNEE TEST -- dpx = arm - reference   [reference: {ref_kind}]")
        print(f"{'arm':<16}{'seed':>5}{'dpx <W':>9}{'dpx >=W':>9}{'shift':>9}")
        for mode in MODES:
            for W in WINDOWS:
                if W == 30 and ref_kind == "W30 same-mode":
                    continue  # would be its own reference
                for s in args.seeds:
                    a = px_of(arm(s, f"W{W}_{mode}"))
                    r = px_of(arm(s, f"W30_{mode}") if ref_kind == "W30 same-mode"
                              else arm(s, "ctrl"))
                    if a is None or r is None:
                        continue
                    d = a - r
                    # Ages below MATURATION_AGE are excluded from "below W": nothing
                    # reproduces there, so selection on them is not the contrast of interest.
                    below = d[MATURATION_AGE:W]
                    above = d[W:]
                    if not len(below) or not len(above):
                        continue
                    print(f"{'W%d_%s' % (W, mode):<16}{s:>5}"
                          f"{below.mean():>9.4f}{above.mean():>9.4f}"
                          f"{above.mean() - below.mean():>9.4f}")

    # ---- per-age difference curves, W=12 (the strongest treatment) ----------
    print("\nPER-AGE dpx, W=12 vs W=30 same-mode   (| marks the window)")
    for mode in MODES:
        rows = []
        for s in args.seeds:
            a, r = px_of(arm(s, f"W12_{mode}")), px_of(arm(s, f"W30_{mode}"))
            if a is not None and r is not None:
                rows.append(a - r)
        if not rows:
            continue
        m = np.mean(rows, axis=0)
        print(f"  {mode} (mean of {len(rows)} seed(s)):")
        for a0 in range(0, len(m), 2):
            bar = "|" if a0 == 12 else " "
            print(f"    age {a0:>2}{bar} {m[a0]:+.4f}  {'#' * int(abs(m[a0]) * 200)}")

    print("\nREAD IT LIKE THIS: 'shift' is the extra deficit above the window relative to")
    print("below it. Negative and consistent across seeds = the horizon eroded survival")
    print("specifically where selection went blind. Compare shift across W: the effect")
    print("should APPEAR AT W in each arm, which no population-size effect can mimic.")


if __name__ == "__main__":
    main()
