"""Neutral clock over generations: the baseline a selected trait stands against.

For each completed run, plots two trajectories across all genotype snapshots:

  load   -- mean ON fraction over all 20 bits x 2 chromatids of the neut loci.
            Phenotypically invisible (single_bit reads only bit 0), so neutral in
            BOTH arms. Relaxes from the G_neut_initgeno=0.5 start to the
            mutation-drift equilibrium 0.1/1.1 = 0.0909. This is the baseline.
  signal -- bit-0 dosage, the one bit the interpreter reads. In MA it is neutral
            and tracks load; in AP it is a driver under selection and peels away.

The vertical gap between signal and load in the AP arm is selection, measured
against each run's own internal neutral ruler.

Reads only completed runs (output_summary.json present and extinct=false); a
still-running run's trajectory is truncated at its latest checkpoint and would
misleadingly look equilibrated-in-progress.

Usage:
    python runs/plot_neut_trajectory.py --datadir runs/ne_ma_ap_data
"""

import argparse
import os
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decode_neut_genotypes import decode, MUT_EQUILIBRIUM, N_NEUT

NAVY = "#003366"
GOLD = "#C9A84C"
ARM_COLOR = {"MA": NAVY, "AP": GOLD}
NE_STYLE = {300: ":", 3000: "--", 30000: "-"}
INIT = 0.5  # G_neut_initgeno


def parse_name(name):
    arm, ne, mode, seed = name.split("_")
    return arm, int(ne[2:]), mode, int(seed[4:])


def load_trajectory(run_dir):
    """(steps, load, signal) across all genotype snapshots, means over pop and loci."""
    gdir = run_dir / "snapshots" / "genotypes"
    snaps = sorted(gdir.glob("*.feather"), key=lambda p: int(p.stem))
    steps, loads, signals = [], [], []
    for s in snaps:
        gdf = pd.read_feather(s)
        if gdf.empty:
            continue
        sig, load = decode(gdf)
        steps.append(int(s.stem))
        loads.append(load.mean())
        signals.append(sig.mean())
    return np.array(steps), np.array(loads), np.array(signals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path(os.environ.get("AEGIS_DATA",
                                        "~/aegis_data/ne_ma_ap_data")).expanduser(),
                   help="run output dir. Kept OUTSIDE Dropbox/git -- simulation output is\n                         GBs. Override with $AEGIS_DATA (e.g. the cluster path).")
    p.add_argument("-o", "--out", default="runs/neut_trajectory.png")
    args = p.parse_args()

    runs = []
    for d in sorted(x for x in args.datadir.iterdir() if x.is_dir()):
        summary = d / "output_summary.json"
        if not summary.exists():
            print(f"  skip {d.name:<28} still running")
            continue
        if json.load(open(summary)).get("extinct"):
            print(f"  SKIP {d.name:<28} EXTINCT")
            continue
        steps, load, signal = load_trajectory(d)
        if len(steps) < 2:
            print(f"  skip {d.name:<28} <2 genotype snapshots (pull them)")
            continue
        arm, ne, _, _ = parse_name(d.name)
        runs.append(dict(name=d.name, arm=arm, ne=ne, steps=steps, load=load, signal=signal))
        print(f"  load {d.name:<28} {len(steps)} snapshots  "
              f"load {load[0]:.3f}->{load[-1]:.3f}  signal {signal[0]:.3f}->{signal[-1]:.3f}")

    if not runs:
        raise SystemExit(f"no completed runs with genotype snapshots in {args.datadir}")
    runs.sort(key=lambda r: (r["arm"], r["ne"]))

    fig, (ax_load, ax_sig) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for r in runs:
        color, ls = ARM_COLOR[r["arm"]], NE_STYLE[r["ne"]]
        label = f"{r['arm']}  Ne={r['ne']:,}"
        ax_load.plot(r["steps"], r["load"], color=color, ls=ls, lw=2, marker="o", ms=3, label=label)
        ax_sig.plot(r["steps"], r["signal"], color=color, ls=ls, lw=2, marker="o", ms=3, label=label)

    for ax, title in [(ax_load, "Neutral load  (all 20 bits — the baseline)"),
                      (ax_sig, "Signal  (bit 0 — selected in AP, neutral in MA)")]:
        ax.axhline(MUT_EQUILIBRIUM, color="crimson", lw=1.2, ls="-", alpha=0.7, zorder=0)
        ax.annotate("mutation–drift equilibrium 0.091", xy=(ax.get_xlim()[1], MUT_EQUILIBRIUM),
                    xytext=(-4, 4), textcoords="offset points", ha="right",
                    fontsize=8, color="crimson")
        ax.axhline(INIT, color="gray", lw=1, ls=":", alpha=0.6, zorder=0)
        ax.annotate("start 0.5", xy=(0, INIT), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color="gray")
        ax.set_xlabel("Simulation step")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    ax_load.set_ylabel("neut ON fraction")

    handles, labels = ax_load.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(runs),
               frameon=False, fontsize=9)
    fig.suptitle(
        "Evolution of the neutral locus over generations (sexual, seed 1)\n"
        "Left: the 19 invisible bits relax to equilibrium in BOTH arms — the neutral baseline.  "
        "Right: bit 0 peels above it under selection in AP.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
