"""Plot age-specific introgression landscape for hybrid runs.

Usage: python runs/plot_introgression.py

Outputs: runs/introgression_landscape.png
"""

import glob
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

NAVY = "#003366"
GOLD = "#C9A84C"
MATURATION_AGE = 5


def load_all_ancestry(run_dir):
    """Load all ancestry snapshots and return (steps, mean_df)."""
    files = sorted(
        glob.glob(str(run_dir / "snapshots" / "ancestry" / "*.csv")),
        key=lambda f: int(pathlib.Path(f).stem),
    )
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def age_profile(df, trait, min_age=0):
    """Mean and std introgression per age class across all steps, normalised by
    per-step population mean (so we see age-specific enrichment/depletion).
    min_age excludes pre-maturation ages for repr."""
    cols = sorted([c for c in df.columns if c.startswith(f"{trait}_")
                   and int(c.split("_")[1]) >= min_age],
                  key=lambda c: int(c.split("_")[1]))
    if not cols:
        return None, None, None
    ages = np.array([int(c.split("_")[1]) for c in cols])
    data = df[cols].values  # (n_steps, n_ages)
    step_mean = data.mean(axis=1, keepdims=True)
    deviation = data - step_mean
    return ages, deviation.mean(axis=0), deviation.std(axis=0)


def trajectory(df, trait, min_age=0):
    """Mean introgression fraction per step (collapsed over age classes >= min_age)."""
    cols = [c for c in df.columns if c.startswith(f"{trait}_")
            and int(c.split("_")[1]) >= min_age]
    if not cols:
        return None, None
    steps = df["step"].values
    means = df[cols].mean(axis=1).values
    return steps, means


def neutral_mean(df):
    """Mean introgression fraction across pre-maturation repr loci (neutral reference)."""
    cols = [c for c in df.columns if c.startswith("repr_")
            and int(c.split("_")[1]) < MATURATION_AGE]
    if not cols:
        return None
    return df[cols].values.mean()


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "Introgression dynamics — pop1 (N=5×10³, navy) vs pop2 (N=5×10², gold)\n"
    "Left: mean introgression over time  |  Right: age-specific enrichment vs population mean",
    fontsize=11,
)

base = pathlib.Path("runs")
runs = {
    "pop1→pop2\n(pop1, N=5×10³ → pop2, N=5×10²)": (base / "hybrid_run_1_2", NAVY),
    "pop2→pop1\n(pop2, N=5×10² → pop1, N=5×10³)": (base / "hybrid_run_2_1", GOLD),
}

for trait, (ax_traj, ax_age) in zip(["surv", "repr"], [(axes[0, 0], axes[0, 1]), (axes[1, 0], axes[1, 1])]):
    for label, (run_dir, color) in runs.items():
        df = load_all_ancestry(run_dir)

        min_age = MATURATION_AGE if trait == "repr" else 0

        # Trajectory panel
        steps, means = trajectory(df, trait, min_age=min_age)
        if steps is not None:
            ax_traj.plot(steps, means, color=color, lw=1.5, label=label.replace("\n", " "))
            ax_traj.axhline(means[0], color=color, lw=0.8, ls=":", alpha=0.5)

        # Age-profile panel
        ages, dev_mean, dev_std = age_profile(df, trait, min_age=min_age)
        if ages is not None:
            smooth = uniform_filter1d(dev_mean, size=3)
            ax_age.plot(ages, smooth, color=color, lw=2, label=label.replace("\n", " "))
            ax_age.fill_between(ages, smooth - dev_std / 2, smooth + dev_std / 2,
                                color=color, alpha=0.15)

    trait_label = "Survival" if trait == "surv" else "Reproduction"

    ax_traj.set_title(f"{trait_label} — mean introgression over time")
    ax_traj.set_xlabel("Simulation step")
    ax_traj.set_ylabel("Mean introgression fraction")
    ax_traj.legend(fontsize=8)
    ax_traj.set_ylim(bottom=0)

    ax_age.axhline(0, color="gray", lw=0.8, ls="--")
    ax_age.axvline(MATURATION_AGE, color="gray", lw=1, ls=":", alpha=0.7, label="Maturation")

    # Neutral reference: mean deviation of pre-maturation repr loci (ages 0..MATURATION_AGE-1)
    for run_dir, color in [(base / "hybrid_run_1_2", NAVY), (base / "hybrid_run_2_1", GOLD)]:
        df_all = load_all_ancestry(run_dir)
        neut_cols = [c for c in df_all.columns if c.startswith("repr_")
                     and int(c.split("_")[1]) < MATURATION_AGE]
        if neut_cols:
            all_cols = [c for c in df_all.columns if c.startswith("surv_") or c.startswith("repr_")]
            step_mean = df_all[all_cols].values.mean(axis=1, keepdims=True)
            neut_dev = (df_all[neut_cols].values - step_mean).mean()
            ax_age.axhline(neut_dev, color=color, lw=1.2, ls="--", alpha=0.7)

    ax_age.set_title(f"{trait_label} — age-specific enrichment (deviation from pop mean)\n"
                     "Dashed = pre-maturation repr neutral reference")
    ax_age.set_xlabel("Age class")
    ax_age.set_ylabel("Introgression − population mean")
    ax_age.legend(fontsize=8)

plt.tight_layout()
outpath = base / "introgression_landscape.png"
plt.savefig(outpath, dpi=150)
print(f"Saved {outpath}")

# Print summary stats
print("\n--- Summary ---")
for label, (run_dir, _) in runs.items():
    df = load_all_ancestry(run_dir)
    first = df[df["step"] == df["step"].min()]
    last = df[df["step"] == df["step"].max()]
    surv_cols = [c for c in df.columns if c.startswith("surv_")]
    print(f"{label.strip()}: surv introgression  start={first[surv_cols].values.mean():.3f}  end={last[surv_cols].values.mean():.3f}")
