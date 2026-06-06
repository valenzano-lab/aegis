"""Animate lattice snapshots from an AEGIS run.

Reads the per-step CSV snapshots in <sim_dir>/lattice/step*.csv and produces
either a PNG montage of selected timesteps or an animated GIF of every
snapshot, color-coded by a chosen attribute.

Usage:
    python runs/lattice_animate.py <sim_dir> [--mode montage|gif] [--color lineage|ancestry|age|sex]

Default: PNG montage of up to 9 evenly-spaced snapshots, colored by lineage.

Output:
    PNG montage  -> <sim_dir>/lattice/montage.png
    Animated GIF -> <sim_dir>/lattice/animation.gif

GIF mode requires `pillow` (which matplotlib already depends on).
"""

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd


def axial_to_xy(q: np.ndarray, r: np.ndarray) -> tuple:
    """Convert hex axial (q, r) to 2D Cartesian (x, y) for plotting.

    Standard "pointy-top" hex orientation:
        x = sqrt(3) * (q + r/2)
        y = 3/2 * r
    Output is unitless; the rendering rescales to fit the figure.
    """
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y


def load_snapshots(lattice_dir: pathlib.Path) -> list:
    """Return a list of (step, dataframe) tuples, sorted by step."""
    files = sorted(
        lattice_dir.glob("step*.csv"),
        key=lambda p: int(re.search(r"step(\d+)\.csv$", p.name).group(1)),
    )
    snapshots = []
    for path in files:
        step = int(re.search(r"step(\d+)\.csv$", path.name).group(1))
        df = pd.read_csv(path)
        snapshots.append((step, df))
    return snapshots


_FOUNDER_LOOKUP_CACHE: dict = {}


def _build_founder_lookup(sim_dir: pathlib.Path) -> dict:
    """Build {lineage_id -> founder_id} by climbing the parent chain in births.csv.

    Returns an empty dict if no births.csv is found (lineage was per-individual
    only; nothing to climb).
    """
    if sim_dir in _FOUNDER_LOOKUP_CACHE:
        return _FOUNDER_LOOKUP_CACHE[sim_dir]
    births_path = sim_dir / "lineage" / "births.csv"
    if not births_path.is_file():
        _FOUNDER_LOOKUP_CACHE[sim_dir] = {}
        return {}
    births = pd.read_csv(births_path)
    parent_of = dict(zip(births["lineage_id"].tolist(),
                         births["parent_lineage_id"].tolist()))
    founder = {}

    def climb(lid):
        if lid in founder:
            return founder[lid]
        path = []
        cur = lid
        while parent_of.get(cur, -1) != -1:
            if cur in founder:
                root = founder[cur]
                for x in path:
                    founder[x] = root
                return root
            path.append(cur)
            cur = parent_of[cur]
        for x in path + [cur]:
            founder[x] = cur
        return cur

    for lid in parent_of:
        climb(lid)
    _FOUNDER_LOOKUP_CACHE[sim_dir] = founder
    return founder


def _color_values(df: pd.DataFrame, color_by: str,
                  founder_lookup: dict = None) -> np.ndarray:
    if color_by == "lineage":
        # Default: per-individual lineage ID (busy, every offspring its own colour).
        return df["lineage_id"].to_numpy()
    if color_by == "founder":
        # Climb parent chain via births.csv → root founder. Surviving cluster
        # then shows which initial founders' descendants persisted.
        if not founder_lookup:
            return df["lineage_id"].to_numpy()  # fall back
        return np.array(
            [founder_lookup.get(int(lid), int(lid)) for lid in df["lineage_id"]]
        )
    if color_by == "ancestry":
        return df["ancestry_fraction"].to_numpy()
    if color_by == "age":
        return df["age"].to_numpy()
    if color_by == "sex":
        return df["sex"].to_numpy()
    raise ValueError(f"unknown color_by: {color_by!r}")


def _color_kwargs(color_by: str, values: np.ndarray) -> dict:
    if color_by in ("age", "sex"):
        return dict(c=values, cmap="viridis")
    if color_by == "ancestry":
        # ancestry fraction is in [0, 1] (or -1 if not tracked)
        return dict(c=values, cmap="coolwarm", vmin=0, vmax=1)
    if color_by in ("lineage", "founder"):
        # Discrete categorical — wrap to tab20 colormap
        return dict(c=values % 20, cmap="tab20")
    raise ValueError(color_by)


def make_montage(snapshots: list, color_by: str, out_path: pathlib.Path,
                 n_panels: int = 9, founder_lookup: dict = None) -> None:
    """Pick up to n_panels evenly-spaced snapshots and tile them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not snapshots:
        print("No snapshots found.")
        return
    n_have = len(snapshots)
    indices = np.linspace(0, n_have - 1, num=min(n_panels, n_have)).astype(int)
    indices = sorted(set(indices.tolist()))
    chosen = [snapshots[i] for i in indices]

    cols = min(3, len(chosen))
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax_idx, (step, df) in enumerate(chosen):
        ax = axes[ax_idx // cols][ax_idx % cols]
        x, y = axial_to_xy(df["q"].to_numpy(), df["r"].to_numpy())
        values = _color_values(df, color_by, founder_lookup)
        kwargs = _color_kwargs(color_by, values)
        ax.scatter(x, y, s=10, **kwargs)
        ax.set_title(f"step {step}  (n={len(df)})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for spare in range(len(chosen), rows * cols):
        axes[spare // cols][spare % cols].axis("off")
    fig.suptitle(f"Lattice — colored by {color_by}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"Wrote {out_path}")


def make_gif(snapshots: list, color_by: str, out_path: pathlib.Path,
             fps: int = 8, founder_lookup: dict = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not snapshots:
        print("No snapshots found.")
        return

    # Pre-compute global bounds so the figure doesn't jitter as cells fill in
    all_q = np.concatenate([df["q"].to_numpy() for _, df in snapshots])
    all_r = np.concatenate([df["r"].to_numpy() for _, df in snapshots])
    x_all, y_all = axial_to_xy(all_q, all_r)
    x_lim = (x_all.min() - 1, x_all.max() + 1)
    y_lim = (y_all.min() - 1, y_all.max() + 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = None

    def init():
        nonlocal scatter
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.scatter([], [], s=12)
        return (scatter,)

    def update(frame_idx):
        step, df = snapshots[frame_idx]
        x, y = axial_to_xy(df["q"].to_numpy(), df["r"].to_numpy())
        values = _color_values(df, color_by, founder_lookup)
        kwargs = _color_kwargs(color_by, values)
        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(x, y, s=12, **kwargs)
        ax.set_title(f"step {step}  (n={len(df)})  [color: {color_by}]")
        return ()

    anim = FuncAnimation(fig, update, frames=len(snapshots), init_func=init,
                         interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sim_dir", type=pathlib.Path,
                        help="AEGIS sim output directory (contains lattice/)")
    parser.add_argument("--mode", choices=("montage", "gif"), default="montage")
    parser.add_argument("--color",
                        choices=("lineage", "founder", "ancestry", "age", "sex"),
                        default="founder")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    lattice_dir = args.sim_dir / "lattice"
    if not lattice_dir.is_dir():
        print(f"No lattice/ subdirectory at {lattice_dir}; was LATTICE_RECORD_RATE > 0?")
        sys.exit(1)

    snapshots = load_snapshots(lattice_dir)
    if not snapshots:
        print(f"No step*.csv files in {lattice_dir}.")
        sys.exit(1)
    print(f"Loaded {len(snapshots)} snapshots from {lattice_dir}")

    founder_lookup = _build_founder_lookup(args.sim_dir) if args.color == "founder" else None
    if founder_lookup is not None and not founder_lookup:
        print("Note: no lineage/births.csv found; founder coloring falls back to per-individual lineage.")

    if args.mode == "montage":
        out = lattice_dir / "montage.png"
        make_montage(snapshots, args.color, out, founder_lookup=founder_lookup)
    else:
        out = lattice_dir / "animation.gif"
        make_gif(snapshots, args.color, out, fps=args.fps, founder_lookup=founder_lookup)


if __name__ == "__main__":
    main()
