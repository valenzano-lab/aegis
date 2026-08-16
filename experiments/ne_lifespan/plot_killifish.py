"""Slide figure: evolved survival against age, one panel per water window.

Each panel contrasts the two generation structures at one window:
  annual   (INCUBATION_PERIOD=-1) -- whole egg bank hatches at the dry-down, synchronous cohort
  overlap  (INCUBATION_PERIOD=3)  -- eggs hatch in waves, age classes coexist
against the no-dry-down control, with the water window marked and everything beyond it
shaded as the region selection cannot see.

FORM. Change over an ordered variable (age) -> lines. Four windows -> small multiples on a
shared y-axis rather than eight series on one plot. Two series per panel, so hues come from
the fixed categorical order; the control is recessive grey because it is a reference, not a
third category.

COLOR. Pastel blue / pastel coral, validated rather than eyeballed: normal-vision OKLab
dE 19.3 (floor 15) and worst-case dichromat dE 13.3 (floor 8, deuteran/protan/tritan all
checked). Pastel means low contrast on a light surface, so the relief rule applies and both
series carry direct labels -- identity never depends on the legend or on colour alone.

Usage:
    python experiments/ne_lifespan/plot_killifish.py --datadir ~/aegis_data/routes --out killifish.png
"""
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore", category=FutureWarning)

WINDOWS = [12, 18, 24, 30]
SEEDS = [1, 2, 3]

# Validated categorical pair (see module docstring) + recessive reference/ink tokens.
BLUE, CORAL = "#7fb3e0", "#f4a582"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE, SHADE = "#fcfcfb", "#f0efeb"
REF = "#b9b8b2"


def px_of(run_dir):
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


def mean_curve(base, arm_fmt, seeds):
    """Mean per-age px across seeds, plus mean e0. None if no seed is present."""
    rows = [px_of(base / arm_fmt.format(s=s)) for s in seeds]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None, None
    m = np.mean(rows, axis=0)
    e0 = float(np.mean([np.cumprod(r).sum() for r in rows]))
    return m, e0


def draw_schematic(ax, anc_e0=None):
    """Horizontal timeline: one pre-evolved ancestor, then allocation to water regimes.

    The design's whole claim rests on a SHARED ancestor -- every regime starts from the
    same equilibrated population, so divergence downstream cannot be burn-in history. The
    schematic exists to make that legible at a glance, so the branch point is the visual
    centre of gravity.

    Dry-downs cannot be drawn to scale: a 200,000-step arm at W=12 contains ~16,700 of
    them. The tick spacing is therefore schematic and ordered by W (dense = dries often),
    and labelled as such rather than implying a literal count.
    """
    BURN_END, TOTAL = 430, 630
    rows = {12: 3, 18: 2, 24: 1, 30: 0}

    ax.add_patch(plt.Rectangle((0, 1.15), BURN_END, 0.7, facecolor=REF, alpha=0.45,
                               edgecolor="none", zorder=2))
    ax.text(BURN_END / 2, 1.5, "pre-evolved ancestor", ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK, zorder=3)
    ax.text(BURN_END / 2, 0.72, "430,000 steps  ·  K = 3,000  ·  permanent water\n"
            "neutral locus equilibrated before any regime is applied",
            ha="center", va="top", fontsize=11, color=INK_SOFT, linespacing=1.5, zorder=3)
    # The starting point every regime splits from -- state it, or the panels' lifespans
    # have nothing to be read against.
    if anc_e0 is not None:
        ax.text(BURN_END / 2, 2.15, f"evolved lifespan  {anc_e0:.1f} steps",
                ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=INK, zorder=3)

    ax.plot([BURN_END, BURN_END], [-0.35, 3.70], color=INK_SOFT, lw=1.3,
            ls=(0, (4, 3)), zorder=4)
    # Each caption gets its own horizontal band -- these three previously overlapped.
    ax.text(BURN_END, 4.75, "pools colonised", ha="center", va="bottom",
            fontsize=11.5, style="italic", color=INK_SOFT)

    for W, r in rows.items():
        y = r * 0.95 - 0.35
        ax.plot([BURN_END, BURN_END + 14], [1.5, y + 0.26], color=REF, lw=1.4, zorder=1)
        ax.add_patch(plt.Rectangle((BURN_END + 14, y), TOTAL - BURN_END - 14, 0.52,
                                   facecolor=SHADE, edgecolor=REF, lw=1.0, zorder=2))
        # Schematic dry-down ticks: spacing ordered by W, not to scale (see docstring).
        x = BURN_END + 14
        while x < TOTAL - 2:
            ax.plot([x, x], [y, y + 0.52], color=BLUE, lw=1.6, alpha=0.75, zorder=3)
            x += W * 1.15
        ax.text(TOTAL + 8, y + 0.26, f"dries every {W} steps", va="center",
                fontsize=11.5, color=INK, zorder=3)

    ax.text(BURN_END + 14 + (TOTAL - BURN_END - 14) / 2, 3.90,
            "200,000 steps per regime  ·  both generation structures  ·  3 seeds",
            ha="center", va="bottom", fontsize=11, color=INK_SOFT)
    ax.text(BURN_END + 14, -0.95, "dry-down ticks schematic, not to scale",
            fontsize=9.5, color=INK_SOFT, style="italic", va="center", ha="left")

    ax.set_xlim(-8, TOTAL + 125)
    ax.set_ylim(-1.3, 5.4)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="~/aegis_data/routes")
    REPO = pathlib.Path(__file__).resolve().parents[2]
    ap.add_argument("--out", default=str(REPO / "runs" / "killifish_windows.png"),
                    help="default: runs/killifish_windows.png, beside the other figures")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    base = pathlib.Path(args.datadir).expanduser()

    anc, anc_e0 = mean_curve(base, "burn_s{s}", args.seeds)   # the shared pre-evolved ancestor
    ctrl, ctrl_e0 = mean_curve(base, "burn_s{s}_K_ctrl", args.seeds)
    if ctrl is None:
        raise SystemExit(f"no control arm found under {base} -- check --datadir")
    ages = np.arange(len(ctrl))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    })
    # Collect every curve first so the shared y-axis is derived, never guessed: the
    # hardcoded floor clipped the annual line off-axis at the longer windows.
    curves = {}
    for W in WINDOWS:
        curves[(W, "annual")] = mean_curve(base, f"burn_s{{s}}_K_W{W}_annual", args.seeds)
        curves[(W, "overlap")] = mean_curve(base, f"burn_s{{s}}_K_W{W}_overlap", args.seeds)
    # Which panel shows the two series furthest apart? That is where direct labels
    # actually identify something.
    seps = {W: (float(np.max(curves[(W, "annual")][0] - curves[(W, "overlap")][0]))
                if curves[(W, "annual")][0] is not None and curves[(W, "overlap")][0] is not None
                else -1) for W in WINDOWS}
    LABEL_PANEL = WINDOWS.index(max(seps, key=seps.get))

    allc = [c for c, _ in curves.values() if c is not None] + [ctrl]
    lo = min(c.min() for c in allc)
    ylo, yhi = lo - 0.06, 1.01

    fig = plt.figure(figsize=(15.5, 7.0), dpi=200)
    gs = fig.add_gridspec(2, len(WINDOWS), height_ratios=[1.0, 2.3], hspace=0.62)
    sch = fig.add_subplot(gs[0, :])
    draw_schematic(sch, anc_e0)
    axes = [fig.add_subplot(gs[1, j]) for j in range(len(WINDOWS))]
    for j, ax in enumerate(axes):
        if j:
            ax.sharey(axes[0])
            ax.tick_params(labelleft=False)

    for i, (ax, W) in enumerate(zip(axes, WINDOWS)):
        # The shadow: every age at or beyond the window is invisible to selection.
        ax.axvspan(W, ages[-1], facecolor=SHADE, edgecolor="none", zorder=0)
        ax.axvline(W, color=INK_SOFT, lw=1.4, ls=(0, (4, 3)), zorder=1)

        ax.plot(ages, ctrl, color=REF, lw=2.0, ls=(0, (2, 2)), zorder=2)

        ann, ann_e0 = curves[(W, "annual")]
        ovl, ovl_e0 = curves[(W, "overlap")]
        if ann is not None:
            ax.plot(ages, ann, color=BLUE, lw=3.4, solid_capstyle="round", zorder=4)
        if ovl is not None:
            ax.plot(ages, ovl, color=CORAL, lw=3.4, solid_capstyle="round", zorder=3)

        ax.set_title(f"Water window = {W} steps\n({W // 6}× age at maturity)",
                     fontsize=14, color=INK, pad=12, linespacing=1.5)
        ax.set_xlabel("Age (steps)", fontsize=13)
        ax.set_xlim(0, ages[-1])
        ax.set_ylim(ylo, yhi)
        ax.set_xticks([0, 6, 12, 18, 24, 30])
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # Relief rule: pastel hues sit under 3:1 on a light surface, so identity is
        # carried by direct labels, not by colour alone. Label in the SHADOW, where the
        # two series are furthest apart -- in the flat pre-window region they overlap.
        if i == 0:
            ax.set_ylabel("Evolved survival per step", fontsize=13.5)
            ax.text(W + 0.5, yhi - 0.012, "selection blind →", color=INK_SOFT, fontsize=11,
                    style="italic", va="top", ha="left", zorder=6,
                    path_effects=[pe.withStroke(linewidth=4, foreground=SHADE)])

        # Direct labels go on whichever panel separates the two series most, found from
        # the data rather than assumed. At the shortest window the two arms are tangled
        # together in the shadow, so labelling there points at nothing.
        if i == LABEL_PANEL and ann is not None and ovl is not None:
            gap = ann - ovl
            x = int(np.argmax(gap))
            halo = [pe.withStroke(linewidth=4, foreground=SURFACE)]
            ax.text(x, ann[x] + 0.012, "annual", color=BLUE, fontsize=13,
                    fontweight="bold", ha="center", va="bottom", path_effects=halo, zorder=6)
            ax.text(x, ovl[x] - 0.014, "overlapping", color=CORAL, fontsize=13,
                    fontweight="bold", ha="center", va="top", path_effects=halo, zorder=6)

        # Headline numbers, in ink rather than series colour, parked top-left where no
        # curve runs (survival is flat and high there in every panel).
        if ann_e0 and ovl_e0:
            txt = f"lifespan   {ann_e0:.1f}  /  {ovl_e0:.1f}"
            if anc_e0 is not None:
                txt += f"        ancestor {anc_e0:.1f}"
            ax.text(0.03, 0.06, txt, transform=ax.transAxes, ha="left",
                    fontsize=11.5, color=INK_SOFT)

    handles = [plt.Line2D([], [], color=BLUE, lw=3.4, label="Non-overlapping (annual)"),
               plt.Line2D([], [], color=CORAL, lw=3.4, label="Overlapping generations"),
               plt.Line2D([], [], color=REF, lw=2.0, ls=(0, (2, 2)), label="No dry-down (control)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035),
               ncol=3, frameon=False, fontsize=13)
    fig.suptitle("Evolved survival collapses exactly where the pool dries",
                 fontsize=18, y=0.99, color=INK)
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE,
                pad_inches=0.35)
    print(f"wrote {args.out}")
    print(f"ancestor e0 = {anc_e0:.2f}" if anc_e0 else
          "ancestor not found -- rsync burn_s*/snapshots/phenotypes/430000.feather")
    print(f"control  e0 = {ctrl_e0:.2f}")


if __name__ == "__main__":
    main()
