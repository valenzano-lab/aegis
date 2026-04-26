"""Equilibrium phenotype comparison: pop1 (N=5×10³) vs pop2 (N=5×10²).

Usage: python runs/plot_equilibrium.py
Output: runs/equilibrium_phenotypes.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NAVY = "#003366"
GOLD = "#C9A84C"
MATURATION_AGE = 5
AGE_LIMIT = 30

base = pathlib.Path("runs")

runs = {
    "pop1  (K=5×10³)": (base / "pop1_equilibrium", NAVY),
    "pop2  (K=5×10²)": (base / "pop2_equilibrium", GOLD),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Equilibrium phenotypes at step 50,000\n"
    "Mean ± SD across individuals. Dashed = maturation age.",
    fontsize=11,
)

trait_labels = {"surv": "Survival probability", "repr": "Reproduction probability"}
trait_cols = {
    "surv": [f"surv_{a}" for a in range(AGE_LIMIT)],
    "repr": [f"repr_{a}" for a in range(MATURATION_AGE, AGE_LIMIT)],
}

for ax, (trait, ylabel) in zip(axes, trait_labels.items()):
    cols = trait_cols[trait]
    ages = np.array([int(c.split("_")[1]) for c in cols])

    for label, (run_dir, color) in runs.items():
        path = run_dir / "snapshots" / "phenotypes" / "50000.feather"
        df = pd.read_feather(path)
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        vals = df[present].values  # (n_individuals, n_ages)
        mean = vals.mean(axis=0)
        sd = vals.std(axis=0)
        ax.plot(ages, mean, color=color, lw=2, label=label)
        ax.fill_between(ages, mean - sd, mean + sd, color=color, alpha=0.15)

    ax.axvline(MATURATION_AGE - 0.5, color="gray", lw=1, ls=":", alpha=0.7,
               label="Maturation")
    ax.set_xlabel("Age class")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

plt.tight_layout()
outpath = base / "equilibrium_phenotypes.png"
plt.savefig(outpath, dpi=150)
print(f"Saved {outpath}")
