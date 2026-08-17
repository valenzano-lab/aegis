"""Three-route decomposition: which path from ecology to life history actually carries the effect?

  Route 1  drift barrier      arm A -- MIGRATION_LONG_RATE sweep, K/N/N·u held IDENTICAL
  Route 2  mutational supply  arm B -- G_muta_initpheno sweep, Ne untouched
  Route 3  extrinsic mortality arm C -- REPRODUCTION_REGULATION off, density feedback on

Every arm branches from the SAME equilibrated ancestor, so between-arm differences cannot be
burn-in history. The question this answers is Ruchitha's: is the lifespan effect attributable
to Ne, or does it run through carrying capacity by some other path?

WHAT TO READ. The headline is not any single arm but the COMPARISON of effect sizes -- how
far each route moves evolved e0 away from `ctrl`. A route that moves it barely at all is not
the mechanism, whatever the theory says.

Arm A additionally reports EARLY vs LATE survival separately. The drift-barrier account is
specific: erosion should concentrate at late ages, where |s| falls under 1/(2Ne). A deficit
spread evenly across ages is some other process.

CAVEAT CARRIED FORWARD. Seeds 2 and 3 branched at 2.5% / 2.9% of the neutral gap remaining
rather than <1% (see HANDOFF.md). Within-seed contrasts are unaffected -- all arms of a seed
share one ancestor -- but arm B is the one place it could bite, since different mutation
rates relax at different speeds. Seed 1 is clean; check it against the others.

Usage:
    python experiments/ne_lifespan/analyze_routes.py --datadir ~/aegis_data/routes
"""
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

MATURATION_AGE = 6
LATE_FROM = 15

# arm -> (label, x-axis value) ; F_ST measured in the lattice calibration, batch 2.
ARM_A = [("A_ld0200", 0.09), ("A_ld0050", 0.17), ("A_ld0010", 0.33), ("A_ld0000", 0.68)]
ARM_B = [("B_mu05", 0.5), ("B_mu20", 2.0), ("B_mu40", 4.0)]
ARM_C = [("C_starv", None)]


def px_of(d):
    d = pathlib.Path(d) / "snapshots" / "phenotypes"
    if not d.is_dir():
        return None
    snaps = sorted(d.glob("*.feather"), key=lambda p: int(p.stem))
    if not snaps:
        return None
    p = pd.read_feather(snaps[-1])
    cols = sorted([c for c in p.columns if c.startswith("surv_")],
                  key=lambda c: int(c.split("_")[1]))
    return p[cols].mean(axis=0).values


def stats(base, arm, seeds):
    out = []
    for s in seeds:
        px = px_of(base / f"burn_s{s}_{arm}")
        if px is None:
            continue
        out.append((float(np.cumprod(px).sum()),
                    float(px[MATURATION_AGE:LATE_FROM].mean()),
                    float(px[LATE_FROM:].mean())))
    if not out:
        return None
    a = np.array(out)
    return dict(n=len(out), e0=a[:, 0].mean(), e0_sd=a[:, 0].std(),
                early=a[:, 1].mean(), late=a[:, 2].mean(), e0_each=a[:, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="~/aegis_data/routes")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    base = pathlib.Path(args.datadir).expanduser()

    ref = stats(base, "ctrl", args.seeds)
    if ref is None:
        raise SystemExit(f"no ctrl arm under {base}")
    print(f"REFERENCE  ctrl (unfragmented, baseline µ, birth-regulated)")
    print(f"  e0 = {ref['e0']:.2f} ± {ref['e0_sd']:.2f}  (n={ref['n']})   "
          f"early px {ref['early']:.4f}   late px {ref['late']:.4f}\n")

    def block(title, arms, xlab):
        print(title)
        print(f"{'arm':<11}{xlab:>8}{'n':>3}{'e0':>8}{'sd':>6}{'Δe0':>8}"
              f"{'early px':>10}{'late px':>9}{'Δlate':>9}")
        rows = []
        for arm, x in arms:
            st = stats(base, arm, args.seeds)
            if st is None:
                print(f"{arm:<11}{'':>8}  -- missing --")
                continue
            rows.append((x, st))
            xs = f"{x:>8.2f}" if x is not None else f"{'—':>8}"
            print(f"{arm:<11}{xs}{st['n']:>3}{st['e0']:>8.2f}{st['e0_sd']:>6.2f}"
                  f"{st['e0'] - ref['e0']:>+8.2f}{st['early']:>10.4f}"
                  f"{st['late']:>9.4f}{st['late'] - ref['late']:>+9.4f}")
        print()
        return rows

    a = block("ROUTE 1 — DRIFT BARRIER   (K, census N and N·u all identical; only structure varies)",
              ARM_A, "F_ST")
    b = block("ROUTE 2 — MUTATIONAL SUPPLY   (Ne untouched; N·u scaled)", ARM_B, "µ ×")
    c = block("ROUTE 3 — EXTRINSIC MORTALITY   (density feedback instead of a birth cap)",
              ARM_C, "")

    # The comparison that decides it: how far does each route move lifespan?
    print("EFFECT SIZE BY ROUTE  (largest |Δe0| from ctrl within each arm)")
    for name, rows in [("1 drift (fragmentation)", a), ("2 supply (mutation rate)", b),
                       ("3 extrinsic mortality", c)]:
        if not rows:
            continue
        worst = max(rows, key=lambda r: abs(r[1]["e0"] - ref["e0"]))
        print(f"  route {name:<26} Δe0 = {worst[1]['e0'] - ref['e0']:+.2f}")

    if a:
        early = [r[1]["early"] - ref["early"] for r in a]
        late = [r[1]["late"] - ref["late"] for r in a]
        print(f"\nROUTE-1 AGE SPECIFICITY   mean Δearly {np.mean(early):+.4f}   "
              f"mean Δlate {np.mean(late):+.4f}")
        print("  The drift barrier predicts erosion concentrated LATE. If Δearly ≈ Δlate the")
        print("  deficit is not age-specific and something other than the barrier is acting.")
        fst = [r[0] for r in a]
        e0s = [r[1]["e0"] for r in a]
        if len(set(fst)) > 2:
            r = np.corrcoef(fst, e0s)[0, 1]
            print(f"\nDOSE–RESPONSE  corr(F_ST, e0) = {r:+.3f} over F_ST {min(fst)}–{max(fst)}")
            print("  A monotone decline across a 7x range of structure is the result;")
            print("  a single arm differing from ctrl is not.")


if __name__ == "__main__":
    main()
