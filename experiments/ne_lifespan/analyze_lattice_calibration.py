"""Analyse the lattice calibration runs: does viscosity buy us Ne range at constant K?

Stdlib-only (yaml used if present, else a minimal line parser for the generated configs)
so it runs anywhere, including a machine with no aegis env.

WHAT IT REPORTS, AND WHY EACH COLUMN IS THERE

  N_mean / N_harm / N_min
      Does census N track K under lattice mode (Q1)? Lattice mode fails a birth when no
      adjacent cell is empty, which is a LOCAL density regulation layered on top of
      REPRODUCTION_REGULATION. If N sits well below K, or the trajectory is unstable, the
      fragmentation arm is not "equal K, varying Ne" and the design has to change.

  occ   occupancy = N / n_cells, against the configured LATTICE_TARGET_DENSITY.

  lin   number of distinct surviving lineages. If this collapses to 1 the concordance
        statistic is undefined -- the run was too long for the metric, not too short.

  P_adj  P(two ADJACENT occupied cells share a lineage)
  P_rnd  P(two RANDOM individuals share a lineage)
  I      (P_adj - P_rnd) / (1 - P_rnd)  -- the isolation-by-distance index.

      I is the headline number. 0 = lineages are spatially unstructured (well mixed);
      1 = neighbours are always same-lineage (complete fragmentation). The normalisation
      MATTERS: fragmented arms lose lineages more slowly, so P_rnd is not comparable
      across arms and the raw P_adj would be misleading. I removes that.

  Ne_glob  genetic Ne from global theta_w, via the units-correct estimator in
      runs/genetic_ne.py. Read it with suspicion: it ASSUMES PANMIXIA. Under
      fragmentation, pooling demes inflates apparent diversity (Wahlund), so global
      theta_w can hold steady or rise while local selection efficiency collapses.
      Ne_glob flat while I climbs is not a null result -- it is the signature, and it is
      the reason a fragmentation sweep cannot be analysed with the panmictic estimator.

Usage:
    python analyze_lattice_calibration.py RUNDIR [RUNDIR ...]
    python analyze_lattice_calibration.py --selftest
"""
import argparse
import csv
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "runs"))
import genetic_ne  # noqa: E402  -- one source of truth for the theta_w units math

# Three of the six hex directions, so each undirected adjacency is counted exactly once.
# Matches the neighbour set in aegis_sim/submodels/lattice.py:
#   (q+1,r) (q-1,r) (q,r+1) (q,r-1) (q+1,r-1) (q-1,r+1)
FORWARD_DIRS = ((1, 0), (0, 1), (1, -1))


def read_config(run_dir):
    """Config lives beside the run dir as <name>.yml. Values we need are flat scalars."""
    p = pathlib.Path(run_dir).with_suffix(".yml")
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(open(p)) or {}
    except ImportError:
        pass
    cfg = {}
    for line in open(p):
        if ":" not in line or line.startswith((" ", "-", "#")):
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        if v in ("true", "false"):
            cfg[k.strip()] = (v == "true")
        else:
            try:
                cfg[k.strip()] = float(v) if ("." in v or "e" in v) else int(v)
            except ValueError:
                cfg[k.strip()] = v
    return cfg


def last_lattice_snapshot(run_dir):
    """Highest-numbered lattice/step{N}.csv -> {(q, r): lineage_id}.

    Returns None if lattice output is absent, which for a LATTICE_MODE=True config is
    the fingerprint of the silent failure described in lattice_calibration_configs.py.
    """
    d = pathlib.Path(run_dir) / "lattice"
    if not d.is_dir():
        return None
    files = sorted(d.glob("step*.csv"), key=lambda p: int(p.stem.replace("step", "")))
    if not files:
        return None
    cells = {}
    with open(files[-1]) as f:
        for row in csv.DictReader(f):
            cells[(int(row["q"]), int(row["r"]))] = int(row["lineage_id"])
    return cells


def concordance(cells):
    """Isolation-by-distance from lineage identity alone.

    Edge cells are simply missing neighbours (no toroidal wraparound reconstructed here
    -- the lattice dimensions are not recorded). That drops a thin boundary of pairs,
    identically for every arm since they share a lattice size, so it cannot bias the
    comparison; it is not corrected for.
    """
    n = len(cells)
    if n < 2:
        return None
    counts = {}
    for lin in cells.values():
        counts[lin] = counts.get(lin, 0) + 1

    same_pairs = tot_pairs = 0
    for (q, r), lin in cells.items():
        for dq, dr in FORWARD_DIRS:
            other = cells.get((q + dq, r + dr))
            if other is None:
                continue
            tot_pairs += 1
            same_pairs += (other == lin)
    if tot_pairs == 0:
        return None

    p_adj = same_pairs / tot_pairs
    # Probability two DISTINCT random individuals share a lineage (sampling w/o replacement).
    p_rnd = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    # Normalised so arms with different lineage diversity stay comparable.
    index = (p_adj - p_rnd) / (1 - p_rnd) if p_rnd < 1 else float("nan")
    return dict(n=n, n_lineages=len(counts), p_adj=p_adj, p_rnd=p_rnd, index=index,
                pairs=tot_pairs)


def block_fst(run_dir, block=10, min_per_block=8):
    """Wright's F_ST between square blocks of the lattice -- the REAL structure metric.

    WHY THIS REPLACED THE LINEAGE METRIC. `lineage_id` is a per-individual pedigree node,
    not a clan label: DEFAULT_PARAMETERS says "each individual is assigned a UNIQUE
    lineage_id at birth and stores the parent's lineage_id". So no two individuals ever
    share one, and any concordance statistic built on it is identically zero whatever the
    spatial model did. Founder clans could be rebuilt by walking parent_lineage_id through
    /lineage/births.csv, but that needs LINEAGE_RATE > 0 and a rerun. Genotypes are
    already on disk and measure the thing we actually care about -- genetic structure,
    not genealogy.

    JOIN VALIDITY. latticerecorder writes one row per individual in population order
    (`for i in range(n)` over the population arrays) and featherrecorder builds its frame
    from the same population without reordering, so row i matches in both. Asserted below
    on the row counts; a mismatch aborts rather than silently pairing wrong individuals.

    F_ST = mean_j Var_b(p_bj) / mean_j pbar_j(1 - pbar_j)   over polymorphic sites j,
    the ratio-of-averages form. Blocks with fewer than `min_per_block` occupants are
    dropped. Small blocks inflate F_ST through sampling noise, but every arm here shares a
    lattice size and census, so the bias is common to all of them and the ACROSS-ARM
    comparison stands; treat the absolute value as an upper bound.
    """
    import warnings
    import numpy as np
    import pandas as pd
    # pandas 3.x routes read_feather through a pyarrow API deprecated in pyarrow 24;
    # it is their internal call, nothing we can do about it, and it buries the table.
    warnings.filterwarnings("ignore", category=FutureWarning,
                            module="pandas.io.feather_format")

    d = pathlib.Path(run_dir)
    snaps = sorted((d / "snapshots" / "genotypes").glob("*.feather"),
                   key=lambda p: int(p.stem))
    lat = sorted((d / "lattice").glob("step*.csv"),
                 key=lambda p: int(p.stem.replace("step", ""))) if (d / "lattice").is_dir() else []
    if not snaps or not lat:
        return None
    if int(snaps[-1].stem) != int(lat[-1].stem.replace("step", "")):
        raise SystemExit(
            f"ABORT {d.name}: last genotype snapshot is step {snaps[-1].stem} but last lattice "
            f"snapshot is step {lat[-1].stem.replace('step','')}. They must be the same step to "
            "be row-aligned.")

    pos = pd.read_csv(lat[-1])
    G = pd.read_feather(snaps[-1]).to_numpy()
    if len(pos) != len(G):
        raise SystemExit(
            f"ABORT {d.name}: lattice snapshot has {len(pos)} rows but genotype snapshot has "
            f"{len(G)}. The row-index join is invalid -- do not interpret.")

    G = G.astype(np.float64)
    blk_q, blk_r = pos["q"].to_numpy() // block, pos["r"].to_numpy() // block
    blk = blk_q * (blk_r.max() + 1) + blk_r
    keep = [b for b in np.unique(blk) if (blk == b).sum() >= min_per_block]
    if len(keep) < 4:
        return None

    pbar = G.mean(axis=0)
    poly = (pbar > 0) & (pbar < 1)
    if not poly.any():
        return None
    P = np.vstack([G[blk == b][:, poly].mean(axis=0) for b in keep])   # (blocks, sites)
    fst = P.var(axis=0, ddof=1).mean() / (pbar[poly] * (1 - pbar[poly])).mean()
    return dict(fst=float(fst), n_blocks=len(keep), n_poly=int(poly.sum()))


def summarize(run_dir):
    name = pathlib.Path(run_dir).name
    cfg = read_config(run_dir)
    row = dict(name=name,
               lattice=bool(cfg.get("LATTICE_MODE", False)),
               mig=cfg.get("MIGRATION_RATE"),
               mig_long=cfg.get("MIGRATION_LONG_RATE"),
               K=cfg.get("RESOURCE_MAXIMUM_AMOUNT"))

    ns = genetic_ne.read_popsize(run_dir)
    if ns:
        row.update(n_mean=statistics.mean(ns), n_harm=statistics.harmonic_mean(ns),
                   n_min=min(ns))

    cells = last_lattice_snapshot(run_dir)
    if row["lattice"] and cells is None:
        raise SystemExit(
            f"ABORT {name}: config sets LATTICE_MODE=True but no lattice/ snapshots exist.\n"
            "  Either LATTICE_RECORD_RATE was 0, or -- the dangerous case -- the population\n"
            "  carried positions=None and every lattice code path silently no-opped, so the\n"
            "  run was WELL MIXED despite its config. That happens when a lattice run is\n"
            "  seeded from a non-lattice pickle (assign_initial_positions is only called by\n"
            "  Population.initialize). Do not interpret this run.")
    if cells:
        c = concordance(cells)
        if c:
            row.update(c)
            # lineage_id is a UNIQUE-PER-INDIVIDUAL pedigree node, not a clan label, so
            # when every individual has its own the statistic is vacuously 0. Say so
            # rather than reporting a confident "no structure".
            if c["n_lineages"] >= 0.99 * c["n"]:
                row["lineage_unusable"] = True
            density = cfg.get("LATTICE_TARGET_DENSITY")
            if density and row.get("n_mean"):
                # n_cells is sized at init from expected capacity / target density.
                row["occ"] = c["n"] / (row["K"] / density) if row.get("K") else None

    if row["lattice"]:
        try:
            f = block_fst(run_dir)
        except ImportError:
            f = None
            row["fst_needs_pandas"] = True
        if f:
            row.update(f)

    rec = genetic_ne.read_simple(run_dir)
    if rec and None not in (rec.get("ne"), rec.get("theta"), rec.get("theta_w")):
        ploidy = genetic_ne.read_ploidy(run_dir)
        L = genetic_ne.read_genome_length(run_dir, ploidy) if ploidy else None
        if L:
            row["ne_glob"] = genetic_ne.genetic_ne(rec["ne"], rec["theta"], rec["theta_w"], L)
    return row


def fmt(v, spec, width):
    return format(v, spec) if isinstance(v, (int, float)) else " " * width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run_dirs:
        ap.error("give run directories, or --selftest")

    rows = [summarize(d) for d in args.run_dirs]
    w = max(len(r["name"]) for r in rows)
    print(f"{'run':<{w}}  {'mig':>6} {'long':>6} {'N_mean':>8} {'N_min':>7} "
          f"{'occ':>5} {'F_ST':>7} {'blk':>4} {'poly':>6} {'Ne_glob':>8}")
    for r in rows:
        print(f"{r['name']:<{w}}  "
              f"{fmt(r.get('mig'), '6.3f', 6)} {fmt(r.get('mig_long'), '6.3f', 6)} "
              f"{fmt(r.get('n_mean'), '8.0f', 8)} "
              f"{fmt(r.get('n_min'), '7.0f', 7)} {fmt(r.get('occ'), '5.2f', 5)} "
              f"{fmt(r.get('fst'), '7.4f', 7)} {fmt(r.get('n_blocks'), '4d', 4)} "
              f"{fmt(r.get('n_poly'), '6d', 6)} {fmt(r.get('ne_glob'), '8.0f', 8)}")

    print()
    verdicts = []
    lat = [r for r in rows if r.get("lattice") and "fst" in r]

    for r in rows:
        if r.get("n_mean") and r.get("K") and r["n_mean"] < 0.75 * r["K"]:
            verdicts.append(
                f"Q1 FAIL  {r['name']}: mean N = {r['n_mean']:.0f} vs K = {r['K']}. Local "
                "placement failure is regulating below K, so this arm is NOT 'equal K'.")
    if not any(v.startswith("Q1 FAIL") for v in verdicts) and rows:
        ns = [r["n_mean"] for r in rows if "n_mean" in r]
        verdicts.append(
            f"Q1 PASS  census N holds at K across every arm (mean {min(ns):.0f}-{max(ns):.0f}). "
            "Lattice local density regulation does NOT depress N, so 'equal K' is intact.")

    if any(r.get("lineage_unusable") for r in rows):
        verdicts.append(
            "note     lineage_id is unique per individual (a pedigree node, not a clan), so the "
            "old concordance index is vacuous and has been dropped. F_ST above replaces it. To "
            "use genealogy instead, set LINEAGE_RATE>0 and walk parent_lineage_id in births.csv.")

    if any(r.get("fst_needs_pandas") for r in rows):
        verdicts.append("F_ST skipped: needs pandas+pyarrow. Use ~/aegis-venv/bin/python.")
    elif len(lat) >= 2:
        lo = min(lat, key=lambda r: r["fst"]); hi = max(lat, key=lambda r: r["fst"])
        spread = hi["fst"] / lo["fst"] if lo["fst"] > 0 else float("inf")
        verdicts.append(
            f"Q2/Q3   F_ST spans {lo['fst']:.4f} ({lo['name']}) .. {hi['fst']:.4f} "
            f"({hi['name']}), a {spread:.1f}x spread. "
            + ("Viscosity IS a structure knob -- proceed, and set the sweep grid from these."
               if spread >= 2 else
               "Too narrow to sweep: viscosity is not buying structure at this run length."))
    elif not lat:
        verdicts.append(
            "Q2/Q3   NO F_ST computed -- genotype snapshots missing. rsync "
            "snapshots/genotypes/ alongside lattice/ and rerun this script.")

    nes = [r["ne_glob"] for r in rows if "ne_glob" in r]
    if nes and lat and max(r["fst"] for r in lat) / max(min(r["fst"] for r in lat), 1e-9) >= 2 \
            and max(nes) < 1.5 * min(nes):
        verdicts.append(
            "Q4      global Ne is flat while F_ST varies -- the Wahlund effect predicted "
            "exactly this. The panmictic estimator cannot measure the fragmentation arm; "
            "the sweep needs within-neighbourhood sampling.")
    for v in verdicts:
        print(" ", v)


def selftest():
    import random
    rows = cols = 20

    # (a) well mixed: lineages assigned at random, so neighbours are no more related
    # than any two individuals. Must give I ~ 0 -- and note this has to be genuine
    # randomness: a deterministic formula like (7q + 13r) % 10 makes every neighbour
    # DIFFER by construction, which is anti-structure (I < 0), not absence of structure.
    rng = random.Random(0)
    mixed = {(q, r): rng.randrange(16) for q in range(rows) for r in range(cols)}
    a = concordance(mixed)

    # (b) fragmented: contiguous 5x5 blocks share a lineage. 16 lineages.
    blocks = {(q, r): (q // 5) * 4 + (r // 5) for q in range(rows) for r in range(cols)}
    b = concordance(blocks)

    # (c) same idea, coarser: 10x10 blocks. 4 lineages -> p_rnd four times higher.
    coarse = {(q, r): (q // 10) * 2 + (r // 10) for q in range(rows) for r in range(cols)}
    c = concordance(coarse)

    assert a["n"] == b["n"] == c["n"] == 400
    assert abs(a["index"]) < 0.06, f"random lattice must show no structure, got I={a['index']}"
    assert b["index"] > 0.7, f"blocked lattice must show strong structure, got I={b['index']}"
    assert c["index"] > 0.7, f"coarse-blocked lattice must too, got I={c['index']}"
    # The point of normalising: p_rnd differs several-fold between (b) and (c) because
    # they hold different numbers of lineages, yet both read as strongly structured.
    # A raw P_adj comparison would not survive that.
    assert c["p_rnd"] > 3 * b["p_rnd"], (b["p_rnd"], c["p_rnd"])
    assert b["p_adj"] > 0.75 and c["p_adj"] > 0.85

    # A single lineage everywhere -> p_rnd == 1 -> index undefined; must not crash.
    one = concordance({(q, r): 0 for q in range(4) for r in range(4)})
    assert one["n_lineages"] == 1 and one["index"] != one["index"], one  # nan

    print(f"selftest OK: random I={a['index']:+.3f} (~0); blocked I={b['index']:.3f} "
          f"(p_rnd {b['p_rnd']:.3f}, {b['n_lineages']} lineages); coarse I={c['index']:.3f} "
          f"(p_rnd {c['p_rnd']:.3f}, {c['n_lineages']} lineages) -> {c['p_rnd']/b['p_rnd']:.1f}x "
          "the background relatedness, both still read as structured, so I is not just "
          "tracking lineage diversity; single-lineage case returns nan without crashing.")


if __name__ == "__main__":
    main()
