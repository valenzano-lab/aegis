"""Population and resource dynamics under constant vs variable resources.

Reproduces the layout of Figure 5A in Sajina & Valenzano 2016 (arXiv:1602.00723):
for each run, N(t) and R(t) with early stages on the left and the last stages on
the right, so that a change in oscillation amplitude over evolutionary time is
visible as a difference between the two panels.

The paper's claim under test:
  variable + large Rbar -> large-amplitude oscillation early, DAMPED by stage 60k
  variable + small Rbar -> oscillation persists undamped
  constant              -> mild fluctuation only

Series read (both written every step, no rate parameter):
  popsize_before_reproduction.csv   N(t), before new offspring are added
  resources_before_scavenging.csv   R(t), the pool individuals draw on that step

Extinct runs are PLOTTED here, not skipped. In this experiment extinction is a
result -- under the drift-plus-filter hypothesis it is the filter itself -- so a
run that dies is data about which dynamics are survivable, not a failed run.

Usage:
    python runs/plot_resource_oscillation.py
    python runs/plot_resource_oscillation.py --datadir ~/aegis_data/resource_oscillation
"""

import argparse
import json
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#003366"
CRIMSON = "#8B2E2E"
EARLY_STAGES = 1500   # left panel width
LATE_STAGES = 1500    # right panel width

# Plot in the paper's order: variable conditions first, constant controls after.
ORDER = ["var_large", "var_large_k1.1", "var_small", "const_large", "const_small"]


def read_series(run_dir):
    """(N, R) as numpy arrays, truncated to their common length."""
    n_path = run_dir / "popsize_before_reproduction.csv"
    r_path = run_dir / "resources_before_scavenging.csv"
    if not (n_path.exists() and r_path.exists()):
        return None, None
    n = pd.read_csv(n_path, header=None).squeeze("columns").to_numpy()
    r = pd.read_csv(r_path, header=None).squeeze("columns").to_numpy()
    m = min(len(n), len(r))
    return n[:m], r[:m]


def amplitude(x, lo, hi):
    """Coefficient of variation over a window -- the damping metric."""
    seg = x[lo:hi]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 2 or seg.mean() == 0:
        return np.nan
    return seg.std() / seg.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path(os.environ.get(
                       "AEGIS_OSC", "~/aegis_data/resource_oscillation")).expanduser())
    p.add_argument("-o", "--out", default="runs/resource_oscillation.png")
    args = p.parse_args()

    runs = []
    for name in ORDER:
        d = args.datadir / name
        if not d.is_dir():
            continue
        n, r = read_series(d)
        if n is None or len(n) == 0:
            print(f"  skip {name:<16} no series yet")
            continue
        summary = d / "output_summary.json"
        extinct = False
        complete = summary.exists()
        if complete:
            extinct = bool(json.load(open(summary)).get("extinct"))
        runs.append(dict(name=name, n=n, r=r, extinct=extinct, complete=complete))
        tag = "EXTINCT" if extinct else ("complete" if complete else "in progress")
        print(f"  {name:<16} {len(n):>6} steps  N {n.min():>6}-{n.max():<6}  "
              f"R {r.min():>7.0f}-{r.max():<9.0f}  {tag}")

    if not runs:
        raise SystemExit(f"no runs with data in {args.datadir}")

    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 2.6 * len(runs)),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06})

    for row, run in enumerate(runs):
        n, r = run["n"], run["r"]
        total = len(n)
        late_start = max(EARLY_STAGES, total - LATE_STAGES)
        windows = [(0, min(EARLY_STAGES, total)), (late_start, total)]

        # Resources can peak orders of magnitude above N (a crashed population lets the
        # pool accumulate unchecked), so they need their own axis or N is invisible.
        nmax = np.nanmax(n) * 1.10
        rfin = r[np.isfinite(r)]
        rmax = (np.nanmax(rfin) if len(rfin) else 1.0) * 1.10

        for col, (lo, hi) in enumerate(windows):
            ax = axes[row][col]
            x = np.arange(lo, hi)
            axr = ax.twinx()
            axr.plot(x, r[lo:hi], color=CRIMSON, lw=0.7, alpha=0.55)
            axr.set_ylim(0, rmax)
            axr.spines[["top", "left"]].set_visible(False)
            axr.tick_params(axis="y", colors=CRIMSON, labelsize=7)
            if col == 0:
                axr.set_yticklabels([])
                axr.spines["right"].set_visible(False)
            ax.plot(x, n[lo:hi], color=NAVY, lw=0.9)
            ax.set_ylim(0, nmax)
            ax.set_xlim(lo, hi if hi > lo else lo + 1)
            ax.set_zorder(axr.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", colors=NAVY, labelsize=7)
            if col == 1:
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
            cv = amplitude(n, lo, hi)
            ax.text(0.98, 0.92, f"CV(N)={cv:.3f}" if np.isfinite(cv) else "CV(N)=n/a",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="gray")

        label = run["name"] + ("  [EXTINCT]" if run["extinct"] else
                               "" if run["complete"] else "  [in progress]")
        axes[row][0].set_ylabel(label, fontsize=9)

    axes[-1][0].set_xlabel("Simulation step (early)")
    axes[-1][1].set_xlabel("Simulation step (late)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=NAVY, lw=1.5), Line2D([], [], color=CRIMSON, lw=1.5, alpha=0.6)],
               ["Population (left axis)", "Resources (right axis)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Population and resource dynamics: constant vs variable resources\n"
        "Left = first 1,500 stages, right = final 1,500. Damping = lower CV(N) on the right.\n"
        "Note the axes differ: resources peak far above N when a crashed population lets the pool accumulate.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}  ({len(runs)} runs)")

    print("\ndamping check -- CV(N) early vs late:")
    for run in runs:
        n, total = run["n"], len(run["n"])
        early = amplitude(n, 0, min(EARLY_STAGES, total))
        late = amplitude(n, max(EARLY_STAGES, total - LATE_STAGES), total)
        verdict = ("damped" if np.isfinite(early) and np.isfinite(late) and late < 0.6 * early
                   else "persists")
        print(f"  {run['name']:<16} early {early:.3f}  late {late:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
