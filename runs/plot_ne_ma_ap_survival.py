"""Survival, mortality and AP driver dynamics for the Ne x {MA, AP} sweep.

Reads the final snapshot of every completed run and plots:

  1. survival probability px  -- per-age survival, the thing that evolves
  2. mortality qx = 1 - px    -- log scale; a straight line here is Gompertz
  3. survivorship lx          -- cumprod(px), the fraction reaching each age
  4. AP driver frequency      -- which pleiotropic drivers selection actually fixed

Panels 1-3 baseline against the step-0 state: flat 0.95 at every age
(G_surv_initgeno=0.833), i.e. the non-aging start. Distance from that grey line
is the aging that evolved.

Panel 4 is the mechanism test. Each AP driver raises surv at one age and lowers
it at another; ap_specs() assigns the (+early,-late) and (-early,+late) quadrants
50/50, so AP is a test rather than an assumption. If antagonistic pleiotropy is
what drives AP's late-life cost, selection should fix the (+early,-late)
"Williams" drivers and purge their mirror images. The MA arm is the control: the
same 50 neut loci exist and mutate identically, but PHENOMAP_SPECS is absent so
they are wired to nothing -- both classes should then sit at the low frequency
that drift under MUTATION_RATIO=0.1 predicts, with no split between them.
The MA classification is therefore counterfactual: it asks what these loci WOULD
have done had they been wired, which is exactly what makes it a control.

Only runs carrying output_summary.json are plotted -- a run still in flight has
a final snapshot that is merely its latest one, which would silently be compared
against completed runs at a different step.

Usage:
    python runs/plot_ne_ma_ap_survival.py --datadir runs/ne_ma_ap_data
    python runs/plot_ne_ma_ap_survival.py            # defaults to the cluster path
"""

import argparse
import os
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ne_ma_ap_configs import ap_specs  # noqa: E402  -- reproduces each seed's AP architecture

AGE_LIMIT = 50
MATURATION_AGE = 10
INIT_SURV = 0.95  # G_surv_initgeno=0.833 -> 0.95 at every age

NAVY = "#003366"
GOLD = "#C9A84C"
CRIMSON = "#8B2E2E"
SLATE = "#8FA3B8"
ARM_COLOR = {"MA": NAVY, "AP": GOLD}
KIND_COLOR = {"williams": CRIMSON, "anti": SLATE}
KIND_LABEL = {"williams": "Williams  (+surv early, −surv late)",
              "anti": "anti-Williams  (−surv early, +surv late)"}

# The recorded neut column is scaled by G_repr_hi (0.5) rather than G_neut_hi (1.0):
# Phenotypes.get_trait_position() lays traits out as index*AGE_LIMIT, assuming every
# trait occupies AGE_LIMIT columns, but non-evolvable traits have length 0. With only
# surv and neut evolvable, neut sits at columns 50:100 -- which that layout calls
# 'repr' -- so repr's [0, 0.5] range is applied to it. Undo it here so the axis reads
# as true dosage (0 = homozygous off, 0.5 = het, 1 = homozygous on).
# This is RECORDING-ONLY: the phenomap acts on the interpretome inside
# architecture.compute(), before this rescale, so driver effects on surv are correct.
NEUT_RECORD_SCALE = 0.5
# Ne -> linestyle: drift strength. Small Ne = most drift = most broken up.
NE_STYLE = {300: ":", 3000: "--", 30000: "-"}


def parse_name(name):
    """AP_Ne3000_sexual_seed1 -> ('AP', 3000, 'sexual', 1)."""
    arm, ne, mode, seed = name.split("_")
    return arm, int(ne[2:]), mode, int(seed[4:])


def classify_drivers(seed):
    """driver index (1-based on the neut loci) -> 'williams' | 'anti', for this seed.

    ap_specs emits two rows per driver: [neut, i, surv, t1, +w] and [..., t2, -w]
    with t1 < t2. A positive first weight is therefore +early/-late = Williams.
    source_index is 1-based (see composite/architecture.py::_build_phenomap), so
    driver i is phenotype column neut_{i-1}.
    """
    specs = ap_specs(seed)
    kinds = {}
    for k in range(0, len(specs), 2):
        _, i, _, _, w1 = specs[k]
        kinds[int(i)] = "williams" if float(w1) > 0 else "anti"
    return kinds


def load_runs(datadir):
    """Return completed runs as a list of dicts, sorted arm-major then Ne."""
    runs = []
    for d in sorted(datadir.iterdir()):
        if not d.is_dir():
            continue
        summary = d / "output_summary.json"
        if not summary.exists():
            print(f"  skip {d.name:<28} still running (no output_summary.json)")
            continue

        with open(summary) as f:
            extinct = json.load(f).get("extinct")
        if extinct:
            # An extinct run is a failure, not a data point -- say so loudly.
            print(f"  SKIP {d.name:<28} EXTINCT -- investigate, do not plot")
            continue

        snaps = sorted(
            (d / "snapshots" / "phenotypes").glob("*.feather"),
            key=lambda p: int(p.stem),
        )
        if not snaps:
            print(f"  skip {d.name:<28} no phenotype snapshots")
            continue

        arm, ne, mode, seed = parse_name(d.name)
        df = pd.read_feather(snaps[-1])
        px = df[[f"surv_{a}" for a in range(AGE_LIMIT)]].values

        # Driver frequencies, if the neut loci were recorded (G_neut_evolvable).
        freqs = {}
        kinds = classify_drivers(seed)
        for i, kind in kinds.items():
            col = f"neut_{i - 1}"
            if col in df.columns:
                freqs.setdefault(kind, []).append(df[col].mean() / NEUT_RECORD_SCALE)

        runs.append(
            dict(
                name=d.name, arm=arm, ne=ne, mode=mode, seed=seed,
                step=int(snaps[-1].stem), n=len(px),
                mean=px.mean(axis=0), sd=px.std(axis=0), freqs=freqs,
            )
        )
        print(f"  load {d.name:<28} step {snaps[-1].stem:>8}  N={len(px)}")

    runs.sort(key=lambda r: (r["arm"], r["ne"]))
    return runs


def plot_life_table(axes, runs, ages):
    for r in runs:
        color, ls = ARM_COLOR[r["arm"]], NE_STYLE[r["ne"]]
        label = f"{r['arm']}  Ne={r['ne']:,}"
        mean, sd = r["mean"], r["sd"]

        axes[0].plot(ages, mean, color=color, ls=ls, lw=2, label=label)
        axes[0].fill_between(ages, mean - sd, mean + sd, color=color, alpha=0.10, lw=0)
        # Clip so px == 1 (zero mortality) does not vanish at log(0).
        axes[1].plot(ages, np.clip(1 - mean, 1e-4, None), color=color, ls=ls, lw=2, label=label)
        axes[2].plot(ages, mean.cumprod(), color=color, ls=ls, lw=2, label=label)

    # Non-aging reference: what every run started as.
    for ax, y in zip(axes, [np.full(AGE_LIMIT, INIT_SURV),
                            np.full(AGE_LIMIT, 1 - INIT_SURV),
                            INIT_SURV ** (ages + 1)]):
        ax.plot(ages, y, color="gray", lw=1.2, alpha=0.6, zorder=0,
                label="Non-aging start (0.95)")

    axes[0].set_ylabel("Survival probability  $p_x$")
    axes[0].set_title("Intrinsic survival")
    axes[0].set_ylim(0.65, 1.005)
    axes[1].set_ylabel("Mortality  $q_x = 1 - p_x$")
    axes[1].set_title("Mortality (log scale; straight = Gompertz)")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("Survivorship  $l_x = \\prod p_x$")
    axes[2].set_title("Survivorship")
    axes[2].set_ylim(0, 1.005)

    for ax in axes:
        ax.set_xlabel("Age class")
        ax.set_xlim(0, AGE_LIMIT - 1)
        ax.spines[["top", "right"]].set_visible(False)
        # After the ylims are final, so the label lands at the top of each panel.
        ax.axvline(MATURATION_AGE, color="gray", lw=1, ls=":", alpha=0.7)
        ax.annotate("maturation", xy=(MATURATION_AGE, ax.get_ylim()[1]),
                    xytext=(3, -4), textcoords="offset points",
                    fontsize=8, color="gray", rotation=90, va="top")


def plot_drivers(ax, runs, rng):
    """Per-driver frequency, split by driver class, one x-position per run."""
    plotted = [r for r in runs if r["freqs"]]
    if not plotted:
        ax.set_visible(False)
        return

    for x, r in enumerate(plotted):
        for dx, kind in [(-0.16, "williams"), (+0.16, "anti")]:
            vals = np.asarray(r["freqs"].get(kind, []))
            if not len(vals):
                continue
            jitter = rng.uniform(-0.05, 0.05, len(vals))
            ax.scatter(np.full(len(vals), x + dx) + jitter, vals,
                       s=14, color=KIND_COLOR[kind], alpha=0.55, lw=0,
                       label=KIND_LABEL[kind] if x == 0 else None, zorder=3)
            ax.hlines(vals.mean(), x + dx - 0.11, x + dx + 0.11,
                      color=KIND_COLOR[kind], lw=2.5, zorder=4)

    ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6, zorder=1)
    ax.annotate("fixed (all homozygous)", xy=(0, 1.0), xytext=(2, 3),
                textcoords="offset points", fontsize=8, color="gray")
    ax.set_xticks(range(len(plotted)))
    ax.set_xticklabels([f"{r['arm']}\nNe={r['ne']:,}" for r in plotted], fontsize=9)
    ax.set_ylabel("Mean driver dosage  (1 = homozygous, 0.5 = het)")
    ax.set_title("Which pleiotropic drivers selection fixed\n"
                 "(MA = control: same loci, wired to nothing)", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for x in range(len(plotted) - 1):
        ax.axvline(x + 0.5, color="gray", lw=0.5, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path(os.environ.get("AEGIS_DATA",
                                        "~/aegis_data/ne_ma_ap_data")).expanduser(),
                   help="run output dir. Kept OUTSIDE Dropbox/git -- simulation output is\n                         GBs. Override with $AEGIS_DATA (e.g. the cluster path).")
    p.add_argument("-o", "--out", default="runs/ne_ma_ap_survival.png")
    args = p.parse_args()

    runs = load_runs(args.datadir)
    if not runs:
        raise SystemExit(f"no completed runs found in {args.datadir}")

    steps = {r["step"] for r in runs}
    if len(steps) > 1:
        print(f"\nWARNING: runs end at different steps {sorted(steps)} -- not comparable")

    ages = np.arange(AGE_LIMIT)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    flat = axes.ravel()

    plot_life_table(flat[:3], runs, ages)
    plot_drivers(flat[3], runs, np.random.default_rng(0))

    handles, labels = flat[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=len(seen), frameon=False, fontsize=9)

    step = sorted(steps)[-1]
    fig.suptitle(
        f"Evolved survival, mortality and AP driver dynamics after {step:,} steps\n"
        "sexual, seed 1; mean ± SD across individuals.  "
        "Colour = arm (MA navy, AP gold); line style = Ne (dotted 300, dashed 3,000, solid 30,000).",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}  ({len(runs)} runs)")

    # Report the effect the driver panel is testing, so the numbers exist outside the plot.
    print("\ndriver dosage (mean, 1=homozygous), williams vs anti:")
    for r in runs:
        if not r["freqs"]:
            continue
        w = np.mean(r["freqs"].get("williams", [np.nan]))
        a = np.mean(r["freqs"].get("anti", [np.nan]))
        print(f"  {r['name']:<26} williams {w:.4f}  anti {a:.4f}  diff {w - a:+.4f}")


if __name__ == "__main__":
    main()
