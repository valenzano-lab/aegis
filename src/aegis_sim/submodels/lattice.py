"""Hexagonal lattice spatial model (opt-in via LATTICE_MODE).

When enabled, individuals occupy positions on a toroidal hexagonal lattice
(parallelogram with wraparound on both axes, axial coordinates (q, r)).
The lattice constrains:

  - mating: females search expanding rings outward for a fertile male
  - offspring placement: random adjacent empty cell, birth fails if none
  - migration: per-step probability of moving to a random adjacent empty
               cell, plus a rare long-distance dispersal to any empty cell

When LATTICE_MODE is False (default), this submodel is not instantiated and
the simulation runs exactly as before. All entry points are no-ops on the
None-positions path; nothing in the bioreactor changes unless the caller
checks LATTICE_MODE first.

Hexagonal geometry: axial coordinates (q, r) with q in [0, ROWS),
r in [0, COLS), toroidal wraparound on both axes. The six neighbors of
(q, r) are:

    ( q+1, r   ), ( q-1, r   ),
    ( q,   r+1 ), ( q,   r-1 ),
    ( q+1, r-1 ), ( q-1, r+1 )

The "ring" of distance d around (q, r) contains 6*d cells (for d >= 1).
Distance 1 is the 6 immediate neighbors; distance 2 is the next 12 cells;
distance 3 is 18; etc.
"""

import logging
from typing import Optional, Tuple

import numpy as np


# Module-level singleton state. Populated by init() when LATTICE_MODE is on.
_state = {
    "rows": 0,
    "cols": 0,
    "occupancy": None,  # 2D int32 array; -1 = empty, otherwise = individual index
    "rng": None,
}


def init(LATTICE_MODE, INITIAL_POPULATION_SIZE, LATTICE_TARGET_DENSITY,
         RESOURCE_MAXIMUM_AMOUNT, rng_seed=None):
    """Initialise the lattice singleton.

    Lattice size is computed from the expected carrying capacity (we use
    RESOURCE_MAXIMUM_AMOUNT as a proxy when available, falling back to
    INITIAL_POPULATION_SIZE) and LATTICE_TARGET_DENSITY:

        n_cells = expected_carrying_capacity / LATTICE_TARGET_DENSITY

    The lattice is then sized as a roughly square parallelogram:
    rows = ceil(sqrt(n_cells)), cols = ceil(n_cells / rows).
    """
    if not LATTICE_MODE:
        # Spatial model disabled. Keep state empty; callers must guard on LATTICE_MODE.
        for k in _state:
            _state[k] = None if k != "rows" and k != "cols" else 0
        return

    expected_capacity = max(int(RESOURCE_MAXIMUM_AMOUNT or 0), int(INITIAL_POPULATION_SIZE))
    n_cells_target = max(1, int(np.ceil(expected_capacity / max(LATTICE_TARGET_DENSITY, 1e-6))))
    rows = max(1, int(np.ceil(np.sqrt(n_cells_target))))
    cols = max(1, int(np.ceil(n_cells_target / rows)))

    _state["rows"] = rows
    _state["cols"] = cols
    _state["occupancy"] = np.full((rows, cols), -1, dtype=np.int32)
    _state["rng"] = np.random.default_rng(rng_seed)

    logging.info(
        "Lattice initialised: %d rows x %d cols = %d cells (target density %.2f, expected capacity %d)",
        rows, cols, rows * cols, LATTICE_TARGET_DENSITY, expected_capacity,
    )


def _wrap(q, r) -> Tuple[int, int]:
    """Toroidal wraparound to canonical (q, r)."""
    return int(q) % _state["rows"], int(r) % _state["cols"]


# The six axial-coordinate offsets to neighbouring hex cells.
_NEIGHBOUR_OFFSETS = np.array(
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)],
    dtype=np.int32,
)


def neighbours(q: int, r: int) -> np.ndarray:
    """Return the (q, r) coordinates of the 6 immediate neighbours of cell (q, r).
    Wrapped onto the torus. Shape: (6, 2)."""
    offsets = _NEIGHBOUR_OFFSETS
    qs = (q + offsets[:, 0]) % _state["rows"]
    rs = (r + offsets[:, 1]) % _state["cols"]
    return np.stack([qs, rs], axis=1)


def ring(q: int, r: int, radius: int) -> np.ndarray:
    """Return all (q, r) cells at hex-distance exactly `radius` from (q, r).

    The ring at distance d has 6*d cells (d >= 1). Wrapped onto the torus.
    Shape: (6 * radius, 2). For radius == 0, returns just (q, r).

    Algorithm: start at corner (q + radius, r - radius) (radius steps in
    direction +1,-1) and walk `radius` cells in each of the six side
    directions in turn. The walking directions form a CW cycle around the
    hex; see _RING_WALK_DIRS below for the order verified against the
    six immediate neighbours at radius=1.
    """
    if radius == 0:
        return np.array([[q % _state["rows"], r % _state["cols"]]], dtype=np.int32)

    cur_q = q + radius
    cur_r = r - radius
    out = np.empty((6 * radius, 2), dtype=np.int32)
    idx = 0
    for dq, dr in _RING_WALK_DIRS:
        for _ in range(radius):
            out[idx, 0] = cur_q % _state["rows"]
            out[idx, 1] = cur_r % _state["cols"]
            idx += 1
            cur_q += dq
            cur_r += dr

    return out


# Walking directions around a hex ring, in CW order starting from the
# corner at (q + radius, r - radius). Verified against immediate neighbours.
_RING_WALK_DIRS = (
    (-1, 0),   # west
    (-1, 1),   # south-west
    (0, 1),    # south
    (1, 0),    # east
    (1, -1),   # north-east
    (0, -1),   # north
)


def is_empty(q: int, r: int) -> bool:
    q, r = _wrap(q, r)
    return _state["occupancy"][q, r] == -1


def occupant(q: int, r: int) -> int:
    """Return the individual index at (q, r), or -1 if empty."""
    q, r = _wrap(q, r)
    return int(_state["occupancy"][q, r])


def claim(q: int, r: int, individual_idx: int) -> None:
    q, r = _wrap(q, r)
    if _state["occupancy"][q, r] != -1:
        raise RuntimeError(
            f"Cannot claim cell ({q}, {r}) for individual {individual_idx}: "
            f"already occupied by {_state['occupancy'][q, r]}"
        )
    _state["occupancy"][q, r] = individual_idx


def vacate(q: int, r: int) -> None:
    q, r = _wrap(q, r)
    _state["occupancy"][q, r] = -1


def random_empty_anywhere() -> Optional[Tuple[int, int]]:
    """Pick a uniformly-random empty cell from the entire lattice. None if full."""
    empties = np.argwhere(_state["occupancy"] == -1)
    if len(empties) == 0:
        return None
    pick = _state["rng"].integers(0, len(empties))
    return int(empties[pick, 0]), int(empties[pick, 1])


def random_empty_adjacent(q: int, r: int) -> Optional[Tuple[int, int]]:
    """Pick a uniformly-random empty cell from the 6 neighbours of (q, r).
    None if all neighbours are occupied."""
    cells = neighbours(q, r)
    empties = [(int(c[0]), int(c[1])) for c in cells if _state["occupancy"][c[0], c[1]] == -1]
    if not empties:
        return None
    return empties[_state["rng"].integers(0, len(empties))]


def assign_initial_positions(n: int) -> np.ndarray:
    """Assign n unique empty cells to the initial population. Returns an
    (n, 2) int32 array of (q, r) coordinates. Cells are claimed in the
    occupancy grid; subsequent attempts to claim them will fail.
    """
    if _state["occupancy"] is None:
        raise RuntimeError("Lattice not initialised; call submodels.lattice.init(...) first")

    rows, cols = _state["rows"], _state["cols"]
    n_cells = rows * cols
    if n > n_cells:
        raise ValueError(
            f"Cannot place {n} individuals on a lattice of {n_cells} cells "
            f"(rows={rows}, cols={cols}). Increase LATTICE_TARGET_DENSITY or "
            f"reduce INITIAL_POPULATION_SIZE."
        )

    # Random subset of empty cells. argwhere is fine for init; for hot-path use
    # the dedicated helpers above.
    empties = np.argwhere(_state["occupancy"] == -1)
    pick = _state["rng"].choice(len(empties), size=n, replace=False)
    chosen = empties[pick].astype(np.int32)
    for i, (q, r) in enumerate(chosen):
        _state["occupancy"][q, r] = i
    return chosen
