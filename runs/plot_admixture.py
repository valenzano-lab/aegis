"""STRUCTURE-style stacked bar admixture plot for hybrid runs.

Usage: python runs/plot_admixture.py
Output: runs/admixture_landscape.png
"""

import glob, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

NAVY = "#003366"
GOLD = "#C9A84C"
GRAY = "#AAAAAA"
MAT  = 5
AGE_LIMIT = 30

base = pathlib.Path("runs")


def load_ancestry(run_dir, step="last"):
    files = sorted(
        glob.glob(str(run_dir / "snapshots" / "ancestry" / "*.csv")),
        key=lambda f: int(pathlib.Path(f).stem),
    )
    if not files:
        return None, None
    first = pd.read_csv(files[0])
    last  = pd.read_csv(files[-1])
    return first, last


def trait_array(df, trait, min_age=0, age_limit=AGE_LIMIT):
    cols = sorted([c for c in df.columns if c.startswith(f"{trait}_")
                   and int(c.split("_")[1]) >= min_age],
                  key=lambda c: int(c.split("_")[1]))
    if not cols:
        return np.full(age_limit - min_age, np.nan)
    return df[cols].values.mean(axis=0)


runs = [
    dict(
        dir=base / "hybrid_run_1_2",
        label="pop1→pop2  (pop1 seeds in pop2 background)",
        n_seeds=50, n_bg=500,
        pop1_size="5×10²", pop2_size="5×10²",
        introg_color=NAVY, bg_color=GOLD,
        introg_label="pop1 (N=5×10³)", bg_label="pop2 (N=5×10²)",
    ),
    dict(
        dir=base / "hybrid_run_2_1",
        label="pop2→pop1  (pop2 seeds in pop1 background)",
        n_seeds=500, n_bg=5000,
        pop1_size="5×10³", pop2_size="5×10²",
        introg_color=GOLD, bg_color=NAVY,
        introg_label="pop2 (N=5×10²)", bg_label="pop1 (N=5×10³)",
    ),
]

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=False)
fig.suptitle(
    "Introgression admixture landscape — locus-level ancestry at step 50,000\n"
    "Stacked bars: ancestry fraction at each locus. "
    "Dashed line = initial introgression fraction. "
    "Shaded = pre-maturation repr (effectively neutral).",
    fontsize=11,
)

gap = 3
repr_ages = np.arange(MAT, AGE_LIMIT)          # post-maturation only
n_repr = len(repr_ages)
x_surv = np.arange(AGE_LIMIT)
x_repr = np.arange(n_repr) + AGE_LIMIT + gap
x_neut = np.array([AGE_LIMIT + n_repr + 2 * gap])

for ax, run in zip(axes, runs):
    first, last = load_ancestry(run["dir"])
    if last is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        continue

    init_frac = run["n_seeds"] / (run["n_seeds"] + run["n_bg"])
    ic, bc = run["introg_color"], run["bg_color"]

    surv_vals = trait_array(last, "surv")
    repr_vals = trait_array(last, "repr", min_age=MAT)   # post-maturation only
    # Neutral reference: mean of pre-maturation repr loci (ages 0..MAT-1)
    neut_cols = [c for c in last.columns if c.startswith("repr_")
                 and int(c.split("_")[1]) < MAT]
    neut_val = float(last[neut_cols].values.mean()) if neut_cols else None

    for x_pos, vals in [(x_surv, surv_vals), (x_repr, repr_vals)]:
        ax.bar(x_pos, vals,       color=ic, width=0.85)
        ax.bar(x_pos, 1 - vals,   color=bc, width=0.85, bottom=vals)

    if neut_val is not None:
        ax.bar(x_neut, [neut_val],     color=ic, width=1.5)
        ax.bar(x_neut, [1 - neut_val], color=bc, width=1.5, bottom=[neut_val])
    else:
        ax.bar(x_neut, [1], color="#DDDDDD", width=1.5)
        ax.text(x_neut[0], 0.5, "neut\n(n/a)", ha="center", va="center",
                fontsize=7, color="#666666")

    ax.axhline(init_frac, color="black", lw=1.2, ls="--", alpha=0.6,
               label=f"Initial fraction ({init_frac:.1%})")

    # maturation marker on survival axis only
    ax.axvline(MAT - 0.5, color=GRAY, lw=0.8, ls=":", alpha=0.8)

    # section labels
    repr_center = x_repr[0] + (x_repr[-1] - x_repr[0]) / 2
    for xc, label in [
        (AGE_LIMIT / 2 - 0.5,   "Survival"),
        (repr_center,            f"Reproduction\n(age {MAT}–{AGE_LIMIT-1})"),
        (float(x_neut[0]),       f"Neutral\n(repr 0–{MAT-1})"),
    ]:
        ax.text(xc, 1.03, label, ha="center", va="bottom", fontsize=9,
                fontweight="bold", transform=ax.get_xaxis_transform())

    # x ticks: surv ages every 5, repr ages every 5 (offset by MAT), blank for neutral
    surv_tick_ages = list(range(0, AGE_LIMIT, 5))
    repr_tick_ages = list(range(MAT, AGE_LIMIT, 5))
    ticks  = surv_tick_ages + \
             [AGE_LIMIT + gap + (a - MAT) for a in repr_tick_ages] + \
             [int(x_neut[0])]
    labels = [str(a) for a in surv_tick_ages] + \
             [str(a) for a in repr_tick_ages] + [""]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-1, x_neut[0] + 2)

    ax.set_ylabel("Ancestry fraction")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_title(run["label"], fontsize=10, loc="left")
    ax.set_xlabel("Age class")

    patch_i = mpatches.Patch(color=ic, label=run["introg_label"])
    patch_b = mpatches.Patch(color=bc, label=run["bg_label"])
    init_line = plt.Line2D([0], [0], color="black", lw=1.2, ls="--",
                           label=f"Initial fraction ({init_frac:.1%})")
    ax.legend(handles=[patch_i, patch_b, init_line], loc="upper right",
              fontsize=8, title="Ancestry", title_fontsize=8)

plt.tight_layout()
outpath = base / "admixture_landscape.png"
plt.savefig(outpath, dpi=150)
print(f"Saved {outpath}")
