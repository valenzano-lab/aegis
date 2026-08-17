"""Slide figure: the three routes from carrying capacity to evolved lifespan.

THE QUESTION (Ruchitha). The original sweep set "Ne" via RESOURCE_MAXIMUM_AMOUNT, i.e. via
carrying capacity — so is the aging pattern driven by K or by Ne? The answer is that K reaches
life history by THREE distinct paths, and the original design moved all three at once:

  1  K -> N -> Ne          drift barrier: can selection SEE a late-acting variant?
  2  K -> N·u              mutational supply: how much raw material arrives?
  3  K -> resource-limited mortality   Williams/Medawar: how steep is the selection gradient?

Every arm branches from ONE equilibrated ancestor, so between-arm differences cannot be
burn-in history. Panels: (1) route 1 alone, at identical K, N and N·u — the arm no comparative
dataset can provide; (2) all three effect sizes side by side; (3) the age profile, which
separates the mechanisms independently of effect size.

COLOR, computed not judged. Pastel blue/coral for the two pastel-requested series plus a DARK
violet third. Three low-chroma hues cannot be separated: five pastel triples were tested and
all failed (worst-pair CVD dE 1.2–4.4 against a floor of 8) because blue and teal converge
under deuteranopia. The kept palette scores normal-vision dE 19.3 and worst-case dichromat
13.3; the binding pair is blue-vs-coral, so the third hue only has to be dark. Checked with a
local Viénot–Brettel–Mollon implementation, which runs stricter than the reference validator —
treated as a conservative bound.
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

MATURATION_AGE, LATE_FROM = 6, 15
R1 = [("A_ld0200", 0.09), ("A_ld0050", 0.17), ("A_ld0010", 0.33), ("A_ld0000", 0.68)]
R2 = [("B_mu05", 0.5), ("B_mu20", 2.0), ("B_mu40", 4.0)]

BLUE, CORAL, VIOLET = "#7fb3e0", "#f4a582", "#4a3f7a"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE, SHADE, REF = "#fcfcfb", "#f0efeb", "#b9b8b2"
ROUTE_COL = {1: BLUE, 2: CORAL, 3: VIOLET}


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


def arm(base, name, seeds):
    """(mean px curve, e0 per seed). None if absent."""
    rows = [px_of(base / f"burn_s{s}_{name}") for s in seeds]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None, None
    return np.mean(rows, axis=0), np.array([np.cumprod(r).sum() for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="~/aegis_data/routes")
    REPO = pathlib.Path(__file__).resolve().parents[2]
    ap.add_argument("--out", default=str(REPO / "runs" / "routes_decomposition.png"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    base = pathlib.Path(args.datadir).expanduser()

    ctrl_px, ctrl_e0 = arm(base, "ctrl", args.seeds)
    if ctrl_px is None:
        raise SystemExit(f"no ctrl arm under {base}")
    ref = ctrl_e0.mean()

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12.5,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    })
    fig = plt.figure(figsize=(15.5, 7.4), dpi=200)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 2.1], hspace=0.55, wspace=0.28)

    # ---- schematic: one ancestor, three routes ----------------------------
    sch = fig.add_subplot(gs[0, :])
    sch.add_patch(plt.Rectangle((0, 1.05), 300, 0.75, facecolor=REF, alpha=0.45,
                                edgecolor="none"))
    sch.text(150, 1.42, "one pre-evolved ancestor", ha="center", va="center",
             fontsize=13, fontweight="bold", color=INK)
    # Keep this inside the bar's width or it runs under the branch line.
    sch.text(150, 0.85, f"430,000 steps · K = 3,000 · lifespan {ref:.1f}",
             ha="center", va="top", fontsize=10.5, color=INK_SOFT)
    sch.plot([300, 300], [-0.55, 2.6], color=INK_SOFT, lw=1.2, ls=(0, (4, 3)))
    # One line per route: two-line subtitles at this spacing ran into each other.
    routes = [(1, "ROUTE 1  drift barrier",
               "vary spatial structure — K, N and N·u held identical"),
              (2, "ROUTE 2  mutational supply",
               "vary mutation rate µ — Ne untouched"),
              (3, "ROUTE 3  extrinsic mortality",
               "starvation deaths — N matched to control")]
    for i, (r, title, sub) in enumerate(routes):
        y = 2.0 - i * 0.95
        sch.plot([300, 322], [1.42, y + 0.16], color=REF, lw=1.3)
        # Box must clear the longest title ("ROUTE 3  extrinsic mortality") with room
        # to spare, or the label overruns it into the description column.
        sch.add_patch(plt.Rectangle((322, y - 0.10), 268, 0.52,
                                    facecolor=ROUTE_COL[r], alpha=0.22, edgecolor="none"))
        sch.text(330, y + 0.16, title, fontsize=11.5, fontweight="bold",
                 color=ROUTE_COL[r] if r != 3 else VIOLET, va="center")
        sch.text(612, y + 0.16, sub, fontsize=10.5, color=INK_SOFT, va="center")
    sch.set_xlim(-10, 1080)
    sch.set_ylim(-0.85, 2.75)
    sch.axis("off")

    # ---- panel 1: route 1 dose-response -----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    xs, means = [], []
    for name, fst in R1:
        _, e0 = arm(base, name, args.seeds)
        if e0 is None:
            continue
        xs.append(fst); means.append(e0.mean())
        ax.plot([fst] * len(e0), e0, "o", color=BLUE, ms=7, alpha=0.55,
                markeredgecolor="none", zorder=3)
    ax.plot(xs, means, "-o", color=BLUE, lw=3.0, ms=10, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(ref, color=REF, lw=2.0, ls=(0, (5, 3)), zorder=2)
    ax.text(0.70, ref + 0.12, "control", color=INK_SOFT, fontsize=10.5, ha="right",
            va="bottom")
    r = np.corrcoef(xs, means)[0, 1]
    ax.text(0.04, 0.06, f"r = {r:+.3f}", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=BLUE)
    ax.set_xlabel("Spatial structure  F$_{ST}$", fontsize=12)
    ax.set_ylabel("Evolved lifespan (steps)", fontsize=12.5)
    ax.set_title("Route 1 — Ne alone\nat identical K, N and N·u", fontsize=13,
                 color=INK, pad=10, linespacing=1.5)

    # ---- panel 2: effect size by route ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    bars = []
    for name, _ in R1:
        _, e0 = arm(base, name, args.seeds)
        if e0 is not None:
            bars.append((1, e0.mean() - ref))
    for name, _ in R2:
        _, e0 = arm(base, name, args.seeds)
        if e0 is not None:
            bars.append((2, e0.mean() - ref))
    _, e0 = arm(base, "C_starv_pen", args.seeds)
    if e0 is not None:
        bars.append((3, e0.mean() - ref))
    for i, (r, d) in enumerate(bars):
        ax.barh(i, d, color=ROUTE_COL[r], height=0.68, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK_SOFT, lw=1.2)
    ax.set_yticks([])
    ax.set_xlabel("Change in evolved lifespan vs control", fontsize=12)
    ax.set_title("All three routes move lifespan\n(each bar = one treatment)",
                 fontsize=13, color=INK, pad=10, linespacing=1.5)
    for r, lab, yy in [(1, "drift", 1.5), (2, "supply", 5.0), (3, "mortality", 7.0)]:
        ax.text(0.5, yy, lab, color=ROUTE_COL[r], fontsize=12, fontweight="bold",
                va="center", path_effects=[pe.withStroke(linewidth=4, foreground=SURFACE)])

    # ---- panel 3: age profile ---------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ages = np.arange(len(ctrl_px))
    for r, name in [(1, "A_ld0000"), (2, "B_mu40"), (3, "C_starv_pen")]:
        px, _ = arm(base, name, args.seeds)
        if px is None:
            continue
        d = px - ctrl_px
        ax.plot(ages, d, color=ROUTE_COL[r], lw=3.2, solid_capstyle="round", zorder=3)
        late = d[LATE_FROM:].mean(); early = d[MATURATION_AGE:LATE_FROM].mean()
        ax.text(29.4, d[-1], f" {abs(late / early):.0f}×", color=ROUTE_COL[r],
                fontsize=12, fontweight="bold", va="center")
    ax.axhline(0, color=INK_SOFT, lw=1.2)
    ax.axvspan(LATE_FROM, ages[-1], facecolor=SHADE, edgecolor="none", zorder=0)
    ax.set_xlim(0, 33)
    ax.set_xticks([0, 6, 12, 18, 24, 30])
    ax.set_xlabel("Age (steps)", fontsize=12)
    ax.set_ylabel("Survival deficit vs control", fontsize=12)
    ax.set_title("Every route hits LATE life\n(× = late deficit ÷ early deficit)",
                 fontsize=13, color=INK, pad=10, linespacing=1.5)

    for a in fig.axes[1:]:
        a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig.suptitle("Carrying capacity reaches lifespan by three separable routes",
                 fontsize=17.5, y=1.0, color=INK)
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.35)
    print(f"wrote {args.out}\ncontrol lifespan = {ref:.2f}")


if __name__ == "__main__":
    main()
