"""+R/-R sweep: does richer inflow stabilize or destabilize an oscillation-adapted population?

The ancestor was burned in UNDER oscillation (regulation off, carryover on, Rbar=2000,
cap=10000) until the neutral locus reached p* -- so it is genetically adapted to crashing
life, and the common-mode late-life adaptation is already absorbed. Each arm then resumes
that ancestor with a different resource LEVEL: Rbar in {500,1000,2000,4000,8000}, cap scaled
5x so the sweep is pure magnitude at fixed dynamical shape. Rbar=2000/cap=10000 is the
ancestor's own regime, the anchor. This is the paper's Rbar (INFLOW) axis -- the axis on
which Sajina & Valenzano 2016 claimed "large Rbar STABILIZES" (Fig 5A).

Competing predictions, now tested on an oscillation-adapted population:
  paper (Fig 5A):        richer Rbar -> amplitude DOWN (stabilizes)
  Rosenzweig-MacArthur:  richer Rbar -> amplitude UP   (paradox of enrichment)
The k x cap scan already found enrichment DEstabilizes on the cap axis; this tests the
inflow axis directly.

Measures per arm over the released phase (steps >= release), aggregated across seeds:
  CV(N) amplitude, period (peak detection), crash depth, min N, and extinction.
Extinct arms are reported, not dropped -- extinction at low Rbar is a result.

Usage:
    python runs/plot_rbar_sweep.py --datadir ~/aegis_data/oscillation_rbar --release 200000
"""

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

NAME_RE = re.compile(r"burnin_osc_R2000_s(?P<seed>\d+)_R(?P<rbar>\d+)$")


def arm_metrics(run_dir, release):
    n_path = run_dir / "popsize_before_reproduction.csv"
    if not n_path.exists():
        return None
    try:
        n = pd.read_csv(n_path, header=None).squeeze("columns").to_numpy(dtype=float)
    except pd.errors.EmptyDataError:
        return None
    if len(n) <= release + 100:
        return None
    seg = n[release:]
    summary = run_dir / "output_summary.json"
    extinct = bool(json.load(open(summary)).get("extinct")) if summary.exists() else None

    alive = seg[seg > 0]
    if len(alive) < 100:
        return dict(extinct=True, cv=np.nan, period=np.nan, depth=np.nan,
                    nmin=0.0, nmean=float(seg.mean()))

    peaks, _ = find_peaks(seg, prominence=seg.std() * 0.5)
    troughs, _ = find_peaks(-seg, prominence=seg.std() * 0.5)
    iv = np.diff(peaks)
    depths = []
    for pk in peaks[:-1]:
        after = troughs[troughs > pk]
        if len(after) and seg[pk] > 0:
            depths.append(1 - seg[after[0]] / seg[pk])
    return dict(
        extinct=extinct,
        cv=seg.std() / seg.mean() if seg.mean() else np.nan,
        period=float(np.median(iv)) if len(iv) else np.nan,
        depth=float(np.median(depths)) if depths else np.nan,
        nmin=float(seg.min()), nmean=float(seg.mean()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path("~/aegis_data/oscillation_rbar").expanduser())
    p.add_argument("--release", type=int, default=200000)
    p.add_argument("-o", "--out", default="runs/rbar_sweep.png")
    args = p.parse_args()

    rows = []
    for d in sorted(x for x in args.datadir.iterdir() if x.is_dir()):
        m = NAME_RE.match(d.name)
        if not m:
            continue
        met = arm_metrics(d, args.release)
        if met is None:
            print(f"  skip {d.name:<34} no released-phase data")
            continue
        rows.append(dict(seed=int(m["seed"]), rbar=int(m["rbar"]), **met))
        tag = "EXTINCT" if met["extinct"] else ""
        print(f"  {d.name:<34} CV {met['cv']:.3f}  period {met['period']:>5.1f}  "
              f"depth {met['depth']:.0%}  minN {met['nmin']:.0f}  {tag}")

    if not rows:
        raise SystemExit(f"no arms with data in {args.datadir}")
    D = pd.DataFrame(rows).sort_values(["rbar", "seed"])

    print("\n=== extinctions by Rbar ===")
    for rbar, g in D.groupby("rbar"):
        s = g.extinct.sum()
        if s:
            print(f"  Rbar {rbar}: {int(s)}/{len(g)} extinct")
    if not D.extinct.any():
        print("  none")

    S = D[~D.extinct.astype(bool)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    NAVY = "#003366"
    for ax, col, ylab, title in [
        (axes[0], "cv", "Amplitude  CV(N)",
         "Enrichment: stabilize (paper) or destabilize (RM)?"),
        (axes[1], "period", "Period (steps)", "Cycle period vs resource level"),
        (axes[2], "depth", "Crash depth", "Crash depth vs resource level"),
    ]:
        g = S.groupby("rbar")[col].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=NAVY,
                    marker="o", ms=6, lw=2, capsize=3)
        ax.axvline(2000, color="gray", ls=":", lw=1, alpha=0.7)
        ax.annotate("ancestor's\nregime", xy=(2000, ax.get_ylim()[0]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7, color="gray")
        ax.set_xscale("log")
        ax.set_xticks(sorted(D.rbar.unique()))
        ax.set_xticklabels(sorted(D.rbar.unique()))
        ax.set_xlabel("Rbar  (resource inflow;  cap = 5x Rbar)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Resource-level (+R/-R) sweep off an OSCILLATION-ADAPTED ancestor "
        f"({D.seed.nunique()} seeds; error bars = SD)\n"
        "the paper's Rbar axis: does richer inflow stabilize (Fig 5A) or destabilize (RM)?",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}  ({len(D)} arms)")

    print("\n=== VERDICT: amplitude vs Rbar ===")
    g = S.groupby("rbar")["cv"].mean()
    print("  CV(N) vs Rbar: " + "  ".join(f"{r}:{v:.3f}" for r, v in g.items()))
    if len(g) > 2:
        # The shape matters, not just the endpoints. Compare the ABOVE-baseline arms to
        # the baseline: if enrichment above the adapted level leaves amplitude flat, that
        # is scale-invariance -- neither the paper's stabilization nor RM enrichment.
        anchor = 2000  # the ancestor's regime
        vals = g.to_dict()
        above = [v for r, v in vals.items() if r >= anchor]
        below = [v for r, v in vals.items() if r < anchor]
        flat_above = (max(above) - min(above)) < 0.1 * np.mean(above)
        if flat_above:
            note = ("SCALE-INVARIANT above the adapted baseline (amplitude flat for "
                    "Rbar >= 2000) -- NOT paper stabilization, NOT RM enrichment. "
                    "The paradox of enrichment seen on the CAP axis is driven by the "
                    "cap:inflow ratio, not resource magnitude.")
            if below and np.mean(below) < min(above) * 0.95:
                note += " Impoverishment below baseline mildly REDUCES amplitude."
        else:
            lo, hi = g.iloc[0], g.iloc[-1]
            note = ("DESTABILIZES (RM)" if hi > lo * 1.1 else
                    "STABILIZES (Fig 5A)" if hi < lo * 0.9 else "flat / non-monotonic")
        print(f"  -> {note}")


if __name__ == "__main__":
    main()
