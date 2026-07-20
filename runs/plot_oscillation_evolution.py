"""How the population dynamics themselves change over evolutionary time.

The Fig 5A claim of Sajina & Valenzano 2016 is that oscillations DAMP as the genome
evolves. Measuring that needs two separate quantities, which turn out to behave
differently here:

  amplitude   CV(N) in a sliding window. "Damping" means this falls.
  regularity  autocorrelation of N at its dominant period, in the same window.
              This says how clock-like the cycle is, independent of how big it is.

A system can lock into a sharp limit cycle (regularity up) without damping at all
(amplitude flat) -- which is what current AEGIS does.

Panel 3 is the phase portrait, N against R. A limit cycle appears as a closed loop;
tightening of the loop from early to late is the same convergence seen in panel 2.

Usage:
    python runs/plot_oscillation_evolution.py
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW = 3000     # sliding window width, ~60 cycles at a period of ~50
STRIDE = 1500
MAX_LAG = 200

COLORS = {
    "var_large": "#8B2E2E",
    "var_small": "#C9A84C",
    "var_large_nomut": "#5B8C5A",
    "const_large": "#003366",
    "const_small": "#8FA3B8",
}
ORDER = ["var_large", "var_large_nomut", "var_small", "const_large", "const_small"]


def read(run_dir):
    n_p = run_dir / "popsize_before_reproduction.csv"
    r_p = run_dir / "resources_before_scavenging.csv"
    if not (n_p.exists() and r_p.exists()):
        return None, None
    n = pd.read_csv(n_p, header=None).squeeze("columns").to_numpy(dtype=float)
    r = pd.read_csv(r_p, header=None).squeeze("columns").to_numpy(dtype=float)
    m = min(len(n), len(r))
    return n[:m], r[:m]


def window_stats(n):
    """(centres, CV, regularity, period) over sliding windows."""
    centres, cvs, regs, pers = [], [], [], []
    for lo in range(0, max(1, len(n) - WINDOW), STRIDE):
        seg = n[lo:lo + WINDOW]
        if len(seg) < WINDOW or seg.mean() == 0:
            continue
        cvs.append(seg.std() / seg.mean())
        s = seg - seg.mean()
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        if ac[0] == 0:
            regs.append(np.nan); pers.append(np.nan)
        else:
            ac = ac / ac[0]
            d = np.diff(ac)
            start = int(np.argmax(d > 0)) if np.any(d > 0) else 1
            seg_ac = ac[start:start + MAX_LAG]
            if len(seg_ac) == 0:
                regs.append(np.nan); pers.append(np.nan)
            else:
                k = int(np.argmax(seg_ac))
                regs.append(seg_ac[k]); pers.append(start + k)
        centres.append(lo + WINDOW // 2)
    return np.array(centres), np.array(cvs), np.array(regs), np.array(pers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path(os.environ.get(
                       "AEGIS_OSC", "~/aegis_data/resource_oscillation")).expanduser())
    p.add_argument("-o", "--out", default="runs/oscillation_evolution.png")
    args = p.parse_args()

    runs = []
    for name in ORDER:
        d = args.datadir / name
        if not d.is_dir():
            continue
        n, r = read(d)
        if n is None or len(n) < WINDOW + STRIDE:
            print(f"  skip {name:<18} too short")
            continue
        c, cv, reg, per = window_stats(n)
        runs.append(dict(name=name, n=n, r=r, c=c, cv=cv, reg=reg, per=per))
        print(f"  {name:<18} {len(n):>6} steps   CV {cv[0]:.3f}->{cv[-1]:.3f}   "
              f"regularity {reg[0]:.2f}->{reg[-1]:.2f}   period {per[0]:.0f}->{per[-1]:.0f}")

    if not runs:
        raise SystemExit(f"no runs with enough data in {args.datadir}")

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.26)
    ax_cv = fig.add_subplot(gs[0, :2])
    ax_reg = fig.add_subplot(gs[1, :2], sharex=ax_cv)
    ax_ph_e = fig.add_subplot(gs[0, 2])
    ax_ph_l = fig.add_subplot(gs[1, 2])

    for run in runs:
        col = COLORS.get(run["name"], "gray")
        ax_cv.plot(run["c"], run["cv"], color=col, lw=1.8, label=run["name"])
        ax_reg.plot(run["c"], run["reg"], color=col, lw=1.8, label=run["name"])

    ax_cv.set_ylabel("Amplitude:  CV(N)")
    ax_cv.set_title("Amplitude does not fall — no damping", fontsize=10, loc="left")
    ax_cv.tick_params(labelbottom=False)
    ax_reg.set_ylabel("Regularity:  autocorr. at period")
    ax_reg.set_xlabel("Simulation step (window centre)")
    ax_reg.set_title("Regularity rises — the cycle becomes clock-like", fontsize=10, loc="left")
    ax_reg.set_ylim(0, 1.02)
    for ax in (ax_cv, ax_reg):
        ax.spines[["top", "right"]].set_visible(False)
    ax_cv.legend(frameon=False, fontsize=8, ncol=2)

    # Phase portraits: the variable runs only -- constant runs are a static blob.
    var_runs = [r for r in runs if r["name"].startswith("var")]
    for ax, when in [(ax_ph_e, "early"), (ax_ph_l, "late")]:
        for run in var_runs:
            n, r = run["n"], run["r"]
            lo, hi = (200, 1200) if when == "early" else (max(0, len(n) - 1200), len(n))
            rr = r[lo:hi]
            finite = np.isfinite(rr)
            ax.plot(n[lo:hi][finite], rr[finite], color=COLORS.get(run["name"], "gray"),
                    lw=0.7, alpha=0.75, label=run["name"])
        ax.set_xlabel("Population N")
        ax.set_ylabel("Resources R")
        ax.set_title(f"Phase portrait, {when}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax_ph_e.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Do the dynamics evolve? Amplitude vs regularity over 60,000 stages\n"
        "The system converges to a sharp limit cycle without reducing its amplitude.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
