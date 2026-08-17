"""What fragmentation actually looks like: genotype painted onto the lattice.

F_ST is a number; this is the thing it measures. Each dot is one individual at its lattice
cell (q, r), coloured by its position on PC1 of the genotype matrix — the same axis a PCA of
real genomes would show. Under free mixing PC1 is spatial noise; under isolation by distance
it forms patches, because neighbours are relatives.

DATA. The lattice calibration (batch 2): MIGRATION_RATE fixed at 0.01, MIGRATION_LONG_RATE
varied, K = 3000 and census N identical in every arm. So the panels differ ONLY in how far
offspring disperse.

THE JOIN. latticerecorder writes one row per individual in population order, and
featherrecorder builds its genotype frame from the same population unreordered, so row i is
the same individual in both. Asserted on row counts and on the step number rather than assumed.

COLOR. PC1 is signed, so this is a DIVERGING scale: two poles with a neutral midpoint, never a
rainbow. The poles reuse the validated blue/coral pair (normal-vision dE 19.3, worst-case
dichromat 13.3); the midpoint is neutral grey. The sign of PC1 is arbitrary — it is flipped per
panel so the larger group is always blue, otherwise panels look different for no reason.
"""
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=FutureWarning)

# arm -> (F_ST measured in the calibration, long-dispersal rate)
ARMS = [("cal_6_ld1000", 0.045, 0.1), ("cal_4_ld0200", 0.092, 0.02),
        ("cal_2_ld0010", 0.334, 0.001), ("cal_1_ld0000", 0.677, 0.0)]

BLUE, CORAL = "#7fb3e0", "#f4a582"
INK, INK_SOFT, SURFACE, MID = "#0b0b0b", "#52514e", "#fcfcfb", "#dcdbd6"
DIVERGING = LinearSegmentedColormap.from_list("pc1", [BLUE, MID, CORAL])


def load(run_dir):
    """(q, r, pc1) for every individual, joined by row index."""
    d = pathlib.Path(run_dir)
    lat = sorted((d / "lattice").glob("step*.csv"),
                 key=lambda p: int(p.stem.replace("step", "")))
    gen = sorted((d / "snapshots" / "genotypes").glob("*.feather"),
                 key=lambda p: int(p.stem))
    if not lat or not gen:
        return None
    lstep = int(lat[-1].stem.replace("step", ""))
    if lstep != int(gen[-1].stem):
        raise SystemExit(f"ABORT {d.name}: lattice step {lstep} != genotype step "
                         f"{gen[-1].stem}; the row join would pair different snapshots.")
    pos = pd.read_csv(lat[-1])
    G = pd.read_feather(gen[-1]).to_numpy().astype(np.float64)
    if len(pos) != len(G):
        raise SystemExit(f"ABORT {d.name}: {len(pos)} lattice rows vs {len(G)} genotype "
                         "rows — the row-index join is invalid.")
    # PC1 by SVD on centred genotypes, restricted to polymorphic sites.
    keep = (G.std(axis=0) > 0)
    X = G[:, keep] - G[:, keep].mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = X @ Vt[0]
    if np.sum(pc1 > 0) < np.sum(pc1 < 0):   # sign of PC1 is arbitrary; fix it
        pc1 = -pc1
    return pos["q"].to_numpy(), pos["r"].to_numpy(), pc1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="~/aegis_data/latcal2")
    REPO = pathlib.Path(__file__).resolve().parents[2]
    ap.add_argument("--out", default=str(REPO / "runs" / "spatial_structure.png"))
    args = ap.parse_args()
    base = pathlib.Path(args.datadir).expanduser()

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "text.color": INK})
    fig, axes = plt.subplots(1, len(ARMS), figsize=(15.5, 4.6), dpi=200)

    for ax, (name, fst, ld) in zip(axes, ARMS):
        got = load(base / name)
        if got is None:
            ax.text(0.5, 0.5, f"{name}\nmissing", ha="center", va="center",
                    transform=ax.transAxes, color=INK_SOFT)
            ax.axis("off")
            continue
        q, r, pc1 = got
        lim = np.percentile(np.abs(pc1), 98)     # symmetric, outlier-robust
        ax.scatter(r, q, c=pc1, cmap=DIVERGING, vmin=-lim, vmax=lim, s=5,
                   linewidths=0, rasterized=True)
        ax.set_title(f"F$_{{ST}}$ = {fst:.3f}\nlong-distance dispersal {ld}",
                     fontsize=13, color=INK, pad=10, linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_color("#e6e5e1")

    axes[0].set_ylabel("lattice position", fontsize=12, color=INK_SOFT)
    fig.suptitle("Same K, same census size — only how far offspring disperse",
                 fontsize=17.5, y=1.06, color=INK)
    fig.text(0.5, -0.06,
             "each dot is one individual at its lattice cell, coloured by PC1 of its genome  ·  "
             "patches mean neighbours are relatives",
             ha="center", fontsize=11.5, color=INK_SOFT)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.35)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
