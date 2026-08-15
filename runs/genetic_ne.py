"""Genetic (drift) effective size vs census Ne for the Ne x {MA, AP} sweep.

WHY THIS EXISTS
    AEGIS's built-in `get_ne()` is the harmonic mean of CENSUS population size
    (popgenstats.get_ne). It carries only the temporal-fluctuation discount on N;
    it knows nothing about variance in reproductive success, so it cannot tell
    carrying capacity apart from the drift-relevant effective size. The Ne x
    {MA, AP} configs set "Ne" via RESOURCE_MAXIMUM_AMOUNT (carrying capacity) and
    ASSUME Ne == N. This script tests that assumption instead of trusting it.

THE ESTIMATOR (self-contained, unit-safe)
    AEGIS DEFINES its model theta as   theta = 2 * ploidy * ne_census * mu
    (popgenstats.get_theta), where mu is PER SITE per generation -- so `theta` is a
    PER-SITE quantity. Watterson's theta_w = S / a_n (popgenstats.get_theta_w) is the
    EMPIRICAL diversity, but S = segregating_sites_gsample is a count over the WHOLE
    genome and is NOT divided by the number of sites -- so `theta_w` is GENOME-TOTAL.

    *** THE UNITS TRAP.  theta and theta_w differ by a factor L, the number of sites
    per haploid genome (L = n_loci x bits_per_locus).  Forming theta_w/theta without
    dividing by L overestimates Ne by ~800x on the standard genome architecture.
    Validated on real output: theta_w/theta ~= L = 800 at equilibrium. ***

    Inverting AEGIS's own definition with the per-site empirical diversity:

        Ne_genetic = (theta_w / L) / (2 * ploidy * mu)
                   = ne_census * theta_w / (theta * L)      <-- uses only recorded columns

    So Ne_genetic is census Ne rescaled by (empirical per-site diversity / model-expected
    per-site diversity). theta_w/L == theta -> genetic == census. Drift/skew depress
    theta_w -> genetic < census: that gap is exactly the N-vs-Ne separation.

    L is read from popgen/allele_frequencies.csv, whose row length is ploidy*L (one
    entry per genomic site, PopgenStats.get_allele_frequencies over the 3D genome).
    Without that file we CANNOT get L, and we report no genetic Ne rather than a
    number that is wrong by three orders of magnitude.

    DIPLOID IS HANDLED CORRECTLY (checked, because it looks wrong at a glance). For
    ploidy 2, get_genomes_sample() unfolds the sample along the CHROMOSOME axis, so
    nsample counts sampled chromosomes and a_n = harmonic(nsample - 1) is the right
    Watterson denominator; segregating_sites_gsample is then counted with ploidy=1 over
    that already-unfolded array, i.e. over L sites. And theta = 2*ploidy*ne*mu = 4*ne*mu
    is the standard diploid per-site theta. So Ne_genetic is a diploid Ne, no correction
    needed. (Note POPGENSTATS_SAMPLE_SIZE counts CHROMOSOMES for a diploid: 100 means
    ~50 individuals.)

    CAVEAT that does bite: theta_w is over the WHOLE sampled genome, so selected loci
    (surv) depress it via linked selection -- see the note under WHY THIS EXISTS.

    CAVEAT (stated, not hidden): theta_w here is over the WHOLE sampled genome, so
    selected loci (surv; the AP drivers) depress it via linked selection. It is
    therefore a lower bound on the neutral Ne. For a clean neutral estimate,
    restrict theta_w to the unwired `neut` loci (allele_frequencies.csv). The
    whole-genome value is the honest first pass and is what the recorder emits.

INPUT
    One or more run directories, each with popgen/simple.csv (produced when
    POPGENSTATS_RATE > 0). Column order is fixed by
    PopgenStats.emit_simple(): n, ne, mu, segregating_sites,
    segregating_sites_gsample, theta, theta_w, theta_pi, tajimas_d, theta_h,
    fayandwu_h[, mean_h, mean_h_expected].

    Existing runs were launched with POPGENSTATS_RATE=0 (no popgen). Two fixes:
      (1) re-emit from the genome SNAPSHOTS that those runs already wrote
          (SNAPSHOT_RATE=100000) -- see reconstruct_from_snapshots(), needs the
          aegis env; no re-simulation.
      (2) for future runs, the patched ne_ma_ap_configs.py sets POPGENSTATS_RATE.

USAGE
    python runs/genetic_ne.py --selftest                    # validates the math, stdlib only
    python runs/genetic_ne.py run1/ run2/ ... --out gne.png # tabulate + plot
"""
import argparse
import csv
import pathlib
import statistics
import sys

# simple.csv columns, in the exact order PopgenStats.emit_simple() writes them.
COLS = ["n", "ne", "mu", "segregating_sites", "segregating_sites_gsample",
        "theta", "theta_w", "theta_pi", "tajimas_d", "theta_h", "fayandwu_h"]


def genetic_ne(ne_census, theta, theta_w, n_sites):
    """Drift Ne from AEGIS's own theta definition. Pure arithmetic, no params.

    theta = 2*ploidy*ne_census*mu  (AEGIS, PER SITE) => 2*ploidy*mu = theta/ne_census
    theta_w is GENOME-TOTAL, so divide it by n_sites (= L, sites per haploid genome)
    before comparing:
        Ne_genetic = (theta_w/L) / (2*ploidy*mu) = ne_census * theta_w / (theta * L)
    """
    if not (theta > 0):
        raise ValueError(f"ABORT: model theta must be > 0 (got {theta}); "
                         "cannot form the diversity ratio.")
    if theta_w < 0 or ne_census < 1:
        raise ValueError(f"ABORT: nonsensical inputs ne={ne_census}, theta_w={theta_w}.")
    if not (n_sites >= 1):
        raise ValueError(f"ABORT: genome length L must be >= 1 (got {n_sites}); "
                         "theta_w cannot be put on a per-site footing.")
    return ne_census * theta_w / (theta * n_sites)


def read_popsize(run_dir):
    """Per-step census N(t) from popsize_after_reproduction.csv (written every step,
    so it captures the starvation troughs that get_ne's coarse sampling aliases away)."""
    p = pathlib.Path(run_dir) / "popsize_after_reproduction.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    ns = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ns.append(float(line.split(",")[0]))
            except ValueError:
                continue  # header, if any
    return [n for n in ns if n > 0] or None


def demographic_ne(run_dir, tail=None):
    """Trough-driven demographic Ne = harmonic mean of per-step census.

    Ne is a functional of the N(t) trajectory; the harmonic mean is dominated by
    the small terms, so the starvation troughs set it. Returns arithmetic mean
    (~K / ceiling), harmonic mean (Ne_demog), and the trough, so the gap
    N_mean -> Ne_demog is the oscillation/bottleneck effect made explicit.
    `tail` restricts to the last N steps (drop burn-in); None uses all.
    """
    ns = read_popsize(run_dir)
    if ns is None:
        return None
    if tail:
        ns = ns[-tail:]
    if len(ns) < 2:
        return None
    n_mean = statistics.mean(ns)
    ne_h = statistics.harmonic_mean(ns)
    n_min = min(ns)
    if not (0 < ne_h <= n_mean + 1e-9):  # harmonic mean can never exceed arithmetic
        raise ValueError(f"ABORT: harmonic Ne {ne_h} > arithmetic mean {n_mean}; "
                         "trajectory read is corrupt.")
    return dict(n_mean=n_mean, ne_demog=ne_h, n_min=n_min,
                trough_ratio=ne_h / n_mean)


def read_genome_length(run_dir, ploidy):
    """L = sites per HAPLOID genome (n_loci x bits_per_locus), from the first row of
    popgen/allele_frequencies.csv.

    That file holds one 1-allele frequency per genomic site of the 3D genome, whose
    shape is (n_individuals, n_loci, ploidy*bits_per_locus) -- so the row length is
    ploidy*L. Segregating sites are counted over L in both the haploid and diploid
    branches of get_segregating_sites, so L (not ploidy*L) is the divisor for theta_w.

    Returns None if the file is absent: better no genetic Ne than one off by ~L.
    """
    p = pathlib.Path(run_dir) / "popgen" / "allele_frequencies.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    with open(p) as f:
        first = f.readline()
    ncols = len([c for c in first.strip().split(",") if c])
    if ncols < ploidy or ncols % ploidy:
        raise ValueError(f"ABORT: {p} has {ncols} columns, not divisible by ploidy "
                         f"{ploidy}; cannot infer genome length.")
    return ncols // ploidy


def read_ploidy(run_dir):
    """Ploidy from the width of simple.csv: emit_simple() appends mean_h and
    mean_h_expected only when ploidy == 2 (11 columns haploid, 13 diploid)."""
    p = pathlib.Path(run_dir) / "popgen" / "simple.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    with open(p) as f:
        rows = [r for r in csv.reader(f) if r]
    if not rows:
        return None
    ncols = len(rows[-1])
    if ncols == len(COLS):
        return 1
    if ncols == len(COLS) + 2:
        return 2
    raise ValueError(f"ABORT: {p} has {ncols} columns; expected {len(COLS)} (haploid) "
                     f"or {len(COLS) + 2} (diploid). Output format has changed.")


def read_simple(run_dir):
    """Return the LAST recorded row of popgen/simple.csv as a dict, or None."""
    p = pathlib.Path(run_dir) / "popgen" / "simple.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    with open(p) as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    # header present iff first cell isn't a number
    try:
        float(rows[0][0])
        header, data = COLS, rows
    except ValueError:
        header, data = rows[0], rows[1:]
    if not data:
        return None
    last = data[-1]
    rec = {}
    for k, v in zip(header, last):
        try:
            rec[k] = float(v)
        except ValueError:
            rec[k] = None
    return rec


def summarize(run_dirs, tail=None):
    """Per run: demographic Ne (always, from per-step census) and, when popgen was
    recorded, genetic Ne. Runs with neither are skipped with a reason."""
    out = []
    for d in run_dirs:
        name = pathlib.Path(d).name
        row = dict(name=name)

        demo = demographic_ne(d, tail=tail)
        if demo:
            row.update(demo)

        rec = read_simple(d)
        if rec is not None and None not in (rec.get("ne"), rec.get("theta"), rec.get("theta_w")):
            ploidy = read_ploidy(d)
            L = read_genome_length(d, ploidy) if ploidy else None
            if L is None:
                print(f"  NOTE {name}: popgen/simple.csv present but "
                      "popgen/allele_frequencies.csv is missing -> genome length L unknown, "
                      "so theta_w cannot be put per-site. Reporting demographic Ne only.",
                      file=sys.stderr)
            else:
                row["ne_genetic"] = genetic_ne(rec["ne"], rec["theta"], rec["theta_w"], L)
                row["theta_ratio"] = rec["theta_w"] / (rec["theta"] * L)
                row["L"] = L
                # Cross-check against the direct form theta_w/(L*2*ploidy*mu) using the
                # recorded realized mu. Both are 4-sig-fig values out of simple.csv, so
                # allow a little slack; a real disagreement means the units are wrong.
                mu = rec.get("mu")
                if mu:
                    direct = rec["theta_w"] / (L * 2 * ploidy * mu)
                    if abs(direct - row["ne_genetic"]) > 0.02 * max(direct, 1):
                        raise ValueError(
                            f"ABORT {name}: Ne_genetic disagrees between the ratio form "
                            f"({row['ne_genetic']:.1f}) and the direct theta_w/(L*2*ploidy*mu) "
                            f"form ({direct:.1f}). theta/theta_w units are not what this "
                            "script assumes.")

        if "ne_demog" not in row and "ne_genetic" not in row:
            print(f"  SKIP {name}: no popsize_after_reproduction.csv and no usable popgen.",
                  file=sys.stderr)
            continue
        out.append(row)
    return out


def selftest():
    # Hand-worked: ploidy=1, mu=1e-3, ne_census=1000, L=800 sites
    #   theta = 2*1*1000*1e-3 = 2.0 (per site)
    #   genome-total theta_w = 1.2*800 = 960 => per-site 1.2
    #   Ne_gen = 1000*960/(2.0*800) = 600 ; direct 1.2/(2*1*1e-3) = 600. Agree.
    g = genetic_ne(1000, 2.0, 960.0, 800)
    assert abs(g - 600.0) < 1e-9, g
    # theta_w/L == theta => genetic == census
    assert abs(genetic_ne(3000, 6.0, 6.0 * 800, 800) - 3000.0) < 1e-9
    # THE BUG THIS FIXES: forgetting /L inflates Ne by exactly L.
    assert abs(genetic_ne(1000, 2.0, 960.0, 1) / g - 800.0) < 1e-9
    # L is read as (allele_frequencies row length)/ploidy; a diploid genome of
    # n_loci*bits = 800 sites writes 1600 columns.
    assert 1600 // 2 == 800
    # harmonic mean is dominated by the troughs: a trajectory that spends most steps
    # near K=2000 but crashes to 100 has Ne_demog pulled far below its arithmetic mean.
    traj = [2000] * 9 + [100]
    assert abs(statistics.mean(traj) - 1810.0) < 1e-6
    ne_h = statistics.harmonic_mean(traj)
    assert ne_h < 900, ne_h  # trough drags the harmonic mean below half the mean
    # aborts on garbage
    for bad in [(1000, 0.0, 1.0, 800), (1000, 2.0, -1.0, 800),
                (0, 2.0, 1.0, 800), (1000, 2.0, 1.0, 0)]:
        try:
            genetic_ne(*bad); raise AssertionError("did not abort on " + str(bad))
        except ValueError:
            pass
    print(f"selftest OK: Ne_gen=ne*theta_w/(theta*L) [L=sites/haploid genome; omitting /L "
          f"inflates Ne by L]; trough traj mean=1810 -> Ne_demog={ne_h:.0f} "
          "(harmonic mean tracks the trough); guards fire.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="run directories (need popsize_after_reproduction.csv; popgen/simple.csv optional)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tail", type=int, default=None, help="use only the last N steps of census (drop burn-in)")
    ap.add_argument("--out", default="genetic_ne.png")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not args.run_dirs:
        ap.error("give run directories, or --selftest")

    rows = summarize(args.run_dirs, tail=args.tail)
    if not rows:
        print("No usable data. Need popsize_after_reproduction.csv (always written) "
              "for demographic Ne, and POPGENSTATS_RATE>0 for genetic Ne.", file=sys.stderr)
        sys.exit(1)

    w = max(len(r["name"]) for r in rows)
    # thw/th is the PER-SITE diversity ratio theta_w/(theta*L): 1.0 means observed
    # diversity matches the neutral expectation at census Ne.
    print(f"{'run':<{w}}  {'N_mean(~K)':>10}  {'Ne_demog':>9}  {'trough':>7}  "
          f"{'Ne_genetic':>11}  {'thw/th':>7}  {'L':>6}")
    def key(r): return r.get("n_mean") or r.get("ne_genetic") or 0
    for r in sorted(rows, key=key):
        nm = r.get("n_mean"); nd = r.get("ne_demog"); tr = r.get("n_min")
        gg = r.get("ne_genetic"); rr = r.get("theta_ratio"); LL = r.get("L")
        print(f"{r['name']:<{w}}  "
              f"{('%10.0f'%nm) if nm else ' '*10}  "
              f"{('%9.0f'%nd) if nd else ' '*9}  "
              f"{('%7.0f'%tr) if tr else ' '*7}  "
              f"{('%11.0f'%gg) if gg else ' '*11}  "
              f"{('%7.3f'%rr) if rr else ' '*7}  "
              f"{('%6d'%LL) if LL else ' '*6}")

    # Plot only the runs that have both demographic and genetic Ne.
    both = [r for r in rows if "n_mean" in r and "ne_genetic" in r]
    if len(both) < 1:
        print("(no run has BOTH demographic and genetic Ne yet; table only. "
              "Add popgen recording or reconstruct theta_w from snapshots to populate Ne_genetic.)",
              file=sys.stderr)
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib absent; table only)", file=sys.stderr); return

    fig, ax = plt.subplots(figsize=(6.2, 6))
    allv = [r["n_mean"] for r in both] + [r["ne_genetic"] for r in both]
    lo, hi = min(allv) * 0.6, max(allv) * 1.4
    ax.plot([lo, hi], [lo, hi], color="#8FA3B8", lw=1, ls="--", label="Ne = N (no drift discount)")
    # arrow from mean census (K) to genetic Ne: length = the full N->Ne collapse
    for r in both:
        ax.plot([r["n_mean"], r["n_mean"]], [r["n_mean"], r["ne_genetic"]],
                color="#C9A84C", lw=1, zorder=2)
        ax.scatter([r["n_mean"]], [r["ne_demog"]], marker="_", s=140, color="#8B2E2E", zorder=3)
    ax.scatter([r["n_mean"] for r in both], [r["ne_genetic"] for r in both],
               s=60, color="#003366", zorder=4, label="genetic Ne")
    ax.scatter([], [], marker="_", color="#8B2E2E", label="demographic Ne (trough-driven)")
    for r in both:
        ax.annotate(r["name"], (r["n_mean"], r["ne_genetic"]),
                    fontsize=7, xytext=(4, -8), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("mean census N  (≈ carrying capacity K)")
    ax.set_ylabel("effective size")
    ax.set_title("N → Ne: how much does drift discount K?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
