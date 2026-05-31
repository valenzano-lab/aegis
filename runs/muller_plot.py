"""Muller plot (clonal expansion / contraction) from AEGIS lineage data.

Reads <sim_dir>/lineage/births.csv and <sim_dir>/lineage/deaths.csv and
produces a stacked-area plot where each band is one founder lineage
(initial-population ancestor) and band thickness over time = number of
alive descendants of that founder.

Usage:
    python runs/muller_plot.py <sim_dir>
        # writes <sim_dir>/lineage/muller.png

This is an opt-in post-processing tool — no parameter, no auto-run.
Requires LINEAGE_TRACING: true and LINEAGE_RATE > 0 in the sim config so
that both births.csv and deaths.csv exist.
"""

import pathlib
import sys

import numpy as np
import pandas as pd


def compute_founders(births: pd.DataFrame) -> dict:
    """Map each lineage_id to its founder (oldest reachable ancestor)."""
    parent_of = dict(zip(births["lineage_id"].tolist(), births["parent_lineage_id"].tolist()))
    founder = {}

    def find_founder(lid):
        if lid in founder:
            return founder[lid]
        chain = []
        cur = lid
        while parent_of.get(cur, -1) != -1:
            if cur in founder:
                # Found a memoized chain — collapse
                root = founder[cur]
                for x in chain:
                    founder[x] = root
                return root
            chain.append(cur)
            cur = parent_of[cur]
        # cur now has parent == -1, so cur is the founder
        for x in chain + [cur]:
            founder[x] = cur
        return cur

    for lid in parent_of:
        find_founder(lid)
    return founder


def main(sim_dir: pathlib.Path) -> int:
    births_path = sim_dir / "lineage" / "births.csv"
    deaths_path = sim_dir / "lineage" / "deaths.csv"
    if not births_path.exists():
        print(f"No births.csv at {births_path}; run with LINEAGE_TRACING + LINEAGE_RATE>0")
        return 1
    births = pd.read_csv(births_path)
    deaths = pd.read_csv(deaths_path) if deaths_path.exists() else pd.DataFrame(columns=["step", "lineage_id", "cause"])

    death_step = dict(zip(deaths["lineage_id"].tolist(), deaths["step"].tolist()))
    birth_step = dict(zip(births["lineage_id"].tolist(), births["step"].tolist()))
    founder = compute_founders(births)

    max_step = int(max(births["step"].max(), deaths["step"].max() if len(deaths) else 0))
    steps = np.arange(0, max_step + 1)

    # Counts table: founder x step
    unique_founders = sorted(set(founder.values()))
    counts = {f: np.zeros(len(steps), dtype=np.int64) for f in unique_founders}

    for lid, b_step in birth_step.items():
        d_step = death_step.get(lid, max_step + 1)  # still alive at end → past max_step
        f = founder[lid]
        # Alive on [b_step, d_step) — exclude the death step itself (died in that step)
        lo = int(b_step)
        hi = min(int(d_step), max_step + 1)
        if hi > lo:
            counts[f][lo:hi] += 1

    # Convert to matrix
    stack = np.stack([counts[f] for f in unique_founders], axis=0)
    totals = stack.sum(axis=0)
    # Relative frequencies (skip steps where total == 0)
    rel = np.where(totals[None, :] > 0, stack / np.maximum(totals[None, :], 1), 0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(unique_founders))]
    ax_abs.stackplot(steps, stack, colors=colors, linewidth=0)
    ax_abs.set_ylabel("alive individuals")
    ax_abs.set_title(f"Muller plot — {len(unique_founders)} founder lineages, asexual clonal dynamics")

    ax_rel.stackplot(steps, rel, colors=colors, linewidth=0)
    ax_rel.set_ylabel("relative frequency")
    ax_rel.set_xlabel("step")
    ax_rel.set_ylim(0, 1)

    fig.tight_layout()
    out_path = sim_dir / "lineage" / "muller.png"
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}  (founders={len(unique_founders)}, max_step={max_step})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runs/muller_plot.py <sim_dir>")
        sys.exit(2)
    sys.exit(main(pathlib.Path(sys.argv[1])))
