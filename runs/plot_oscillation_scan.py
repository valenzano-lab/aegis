"""k x cap scan: is the cycle demographic or consumer-resource, and does enrichment destabilize?

Every arm resumes the SAME equilibrated burn-in (neutral locus at
MUTATION_RATIO/(1+MUTATION_RATIO); see check_equilibration.py), with regulation released
and the resource dynamics switched on, so differences between arms are caused by the
regime and not by burn-in history.

Two competing predictions are under test:

  k = 1 (mult 0)   Donor-controlled: resources are a constant inflow that does not
                   reproduce. Such systems do not generate consumer-resource cycles; the
                   oscillation seen locally was a COHORT/GENERATION cycle, whose period
                   tracked the demographic timescale (~45 steps at AGE_LIMIT 50 /
                   MATURATION 10, ~99 at 70/16).
  k > 1            Resources self-reproduce, logistically under a cap: Rosenzweig-
                   MacArthur. If a true consumer-resource cycle takes over, the period
                   should DECOUPLE from the demographic timescale.
  raising the cap  RM predicts the PARADOX OF ENRICHMENT -- a larger cap destabilizes and
                   grows the limit cycle. This is the opposite of Sajina & Valenzano 2016,
                   where large Rbar stabilizes. The grid distinguishes them.

Measures per arm, over the released phase only:
  CV(N)        amplitude
  period       median inter-peak interval (peak detection, robust to harmonics --
               autocorrelation picks the second harmonic about as often as the fundamental)
  CV(period)   regularity: low = metronome, high = erratic
  crash depth  median peak-to-trough drop, the quantity that drives extinction risk
  extinct      whether the arm died -- a result here, not a failure

Usage:
    python runs/plot_oscillation_scan.py --datadir ~/aegis_data/oscillation_scan --release 200000
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

NAME_RE = re.compile(r"burnin_R(?P<rbar>\d+)_s(?P<seed>\d+)_k(?P<k>[\d.]+)_cap(?P<cap>\d+)$")


def arm_metrics(run_dir, release):
    n_path = run_dir / "popsize_before_reproduction.csv"
    if not n_path.exists():
        return None
    try:
        n = pd.read_csv(n_path, header=None).squeeze("columns").to_numpy(dtype=float)
    except pd.errors.EmptyDataError:
        print(f"  WARN {run_dir.name:<40} popsize file is EMPTY -- transfer or run failure")
        return None
    if len(n) <= release + 100:
        print(f"  ({run_dir.name}: {len(n):,} rows, released phase needs > {release:,})")
        return None
    seg = n[release:]

    summary = run_dir / "output_summary.json"
    extinct = bool(json.load(open(summary)).get("extinct")) if summary.exists() else None

    alive = seg[seg > 0]
    if len(alive) < 100:
        return dict(extinct=True, cv=np.nan, period=np.nan, period_cv=np.nan,
                    depth=np.nan, nmin=0, nmax=float(seg.max()), steps=len(seg))

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
        period_cv=float(iv.std() / iv.mean()) if len(iv) > 1 and iv.mean() else np.nan,
        depth=float(np.median(depths)) if depths else np.nan,
        nmin=float(seg.min()), nmax=float(seg.max()), steps=len(seg),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path("~/aegis_data/oscillation_scan").expanduser())
    p.add_argument("--release", type=int, default=200000)
    p.add_argument("-o", "--out", default="runs/oscillation_scan.png")
    p.add_argument("--demographic-period", type=float, default=45.0,
                   help="cohort-cycle period measured under donor control; the reference "
                        "the k>1 arms are tested against")
    args = p.parse_args()

    rows = []
    for d in sorted(x for x in args.datadir.iterdir() if x.is_dir()):
        m = NAME_RE.match(d.name)
        if not m:
            continue
        met = arm_metrics(d, args.release)
        if met is None:
            print(f"  skip {d.name:<40} no released-phase data yet")
            continue
        rows.append(dict(seed=int(m["seed"]), k=float(m["k"]), cap=int(m["cap"]), **met))
        print(f"  {d.name:<40} CV {met['cv']:.3f}  period {met['period']:>6.1f}  "
              f"regularity(CVper) {met['period_cv']:.3f}  depth {met['depth']:.0%}"
              + ("  EXTINCT" if met["extinct"] else ""))

    if not rows:
        raise SystemExit(f"no completed arms in {args.datadir}")
    D = pd.DataFrame(rows).sort_values(["k", "cap", "seed"])

    print("\n=== extinctions ===")
    ext = D.groupby(["k", "cap"])["extinct"].agg(["sum", "count"])
    for (k, cap), r in ext.iterrows():
        if r["sum"]:
            print(f"  k={k} cap={cap}: {int(r['sum'])}/{int(r['count'])} extinct")
    if not ext["sum"].any():
        print("  none")

    S = D[~D.extinct.astype(bool)] if D.extinct.notna().any() else D
    caps = sorted(D.cap.unique())
    ks = sorted(D.k.unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(caps)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [
        ("cv", "Amplitude  CV(N)", "Does enrichment destabilize?\nRM: larger cap -> larger cycle"),
        ("period", "Period (steps)", "Demographic or consumer-resource?\nDecoupling from the dashed line = RM"),
        ("period_cv", "Irregularity  CV(period)", "Regularity of the cycle\nlow = metronome"),
    ]
    for ax, (col, ylab, title) in zip(axes, panels):
        for c, cap in zip(colors, caps):
            g = S[S.cap == cap].groupby("k")[col].agg(["mean", "std", "count"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=c, marker="o",
                        ms=5, lw=1.8, capsize=3, label=f"cap = {cap:,}")
        if col == "period":
            ax.axhline(args.demographic_period, color="crimson", ls="--", lw=1.2, alpha=0.8)
            ax.annotate("demographic (cohort) period", xy=(ks[-1], args.demographic_period),
                        xytext=(-4, 5), textcoords="offset points", ha="right",
                        fontsize=8, color="crimson")
        ax.set_xlabel("k  (1 = donor-controlled, >1 = resources self-reproduce)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)

    n_seeds = D.seed.nunique()
    fig.suptitle(
        f"Resource-regime scan off an equilibrated ancestor "
        f"({n_seeds} seeds, released phase only; error bars = SD across seeds)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}  ({len(D)} arms)")

    print("\n=== does the period decouple from demography as k rises? ===")
    for cap in caps:
        g = S[S.cap == cap].groupby("k")["period"].mean()
        if len(g):
            print(f"  cap {cap:>6,}: " + "  ".join(f"k={k}:{v:.0f}" for k, v in g.items()))
    print(f"  (donor-controlled demographic reference: {args.demographic_period:.0f} steps)")

    print("\n=== does raising the cap destabilize (paradox of enrichment)? ===")
    for k in ks:
        g = S[S.k == k].groupby("cap")["cv"].mean()
        if len(g) > 1:
            trend = "DESTABILIZES" if g.iloc[-1] > g.iloc[0] * 1.1 else (
                "stabilizes" if g.iloc[-1] < g.iloc[0] * 0.9 else "flat")
            print(f"  k={k}: CV " + " -> ".join(f"{v:.3f}" for v in g) + f"   {trend}")


if __name__ == "__main__":
    main()
