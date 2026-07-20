"""Decode neut-locus mutational state from genotype snapshots.

The phenotype snapshot records only what the interpreter reads: neut uses the
single_bit interpreter, so only bit 0 of each 20-bit locus reaches neut_*. The
genotype snapshot keeps all 20 bits of both chromatids, i.e. the full mutational
record -- the thing you want for a neutral clock. This module maps genotype
columns back to neut loci and extracts two views:

  signal  -- bit-0 dosage, the same quantity the phenotype records (0 = homozygous
             off, 0.5 = het, 1 = homozygous on). Reproduces neut_* exactly; used
             here only to validate the column mapping.
  load    -- mean ON fraction across ALL 20 bits and both chromatids. This is the
             real neutral baseline: 19 of 20 bits are phenotypically invisible and
             evolve under mutation + drift alone, with no selection, in EITHER arm.
             At MUTATION_RATIO=0.1 its equilibrium is 0.1/1.1 = 0.0909.

Column layout (verified against Genomes.flatten and CompositeArchitecture):
    genomes.array shape is (individuals, ploidy, n_loci, bits); flatten() is a bare
    reshape, so column = chromatid*(n_loci*bits) + physical_locus*bits + bit.
    Loci are stored in PHYSICAL order; logical locus i sits at physical position
    locus_permutation[i], where locus_permutation = default_rng(0).permutation(n_loci).
    Logical order is trait x age: with surv (50) then neut (50) evolvable, logical
    neut locus j is index 50 + j.

Usage:
    python runs/decode_neut_genotypes.py --validate            # self-check on local probe
    python runs/decode_neut_genotypes.py --datadir runs/ne_ma_ap_data
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd

BITS = 20
N_SURV = 50          # surv occupies logical loci 0..49
N_NEUT = 50          # neut occupies logical loci 50..99
N_LOCI = N_SURV + N_NEUT
PLOIDY = 2
MUT_EQUILIBRIUM = 0.1 / 1.1  # neutral ON fraction at MUTATION_RATIO=0.1


def neut_physical_positions():
    """Physical storage index of each logical neut locus (j = 0..49)."""
    # Fixed seed 0: every population shares this layout (see CompositeArchitecture).
    perm = np.random.default_rng(0).permutation(N_LOCI)
    return perm[N_SURV:N_SURV + N_NEUT]


def decode(genotype_df):
    """Return (signal, load), each shape (n_individuals, N_NEUT).

    signal[k, j] = bit-0 dosage of neut locus j in individual k  (matches neut_j)
    load[k, j]   = mean ON fraction over all 20 bits x 2 chromatids of neut locus j
    """
    g = genotype_df.values.astype(np.float32)  # (n_ind, ploidy*n_loci*bits)
    n = len(g)
    g = g.reshape(n, PLOIDY, N_LOCI, BITS)      # undo the flatten
    phys = neut_physical_positions()
    neut = g[:, :, phys, :]                      # (n_ind, ploidy, N_NEUT, bits)

    # signal: collapse the two chromatids' bit 0 the way ploider does
    # (homozygous on -> 1, het -> 0.5, homozygous off -> 0) == mean over chromatids.
    signal = neut[:, :, :, 0].mean(axis=1)       # (n_ind, N_NEUT)
    # load: every bit, both chromatids -- the full mutational record.
    load = neut.mean(axis=(1, 3))                # (n_ind, N_NEUT)
    return signal, load


def validate(probe_dir):
    """Check the decoded bit-0 signal reproduces the recorded neut_* phenotype."""
    gdir = probe_dir / "snapshots" / "genotypes"
    pdir = probe_dir / "snapshots" / "phenotypes"
    steps = sorted(int(p.stem) for p in gdir.glob("*.feather"))
    ok = True
    for step in steps:
        gdf = pd.read_feather(gdir / f"{step}.feather")
        pdf = pd.read_feather(pdir / f"{step}.feather")
        if gdf.empty or pdf.empty:
            continue
        signal, _ = decode(gdf)
        recorded = pdf[[f"neut_{j}" for j in range(N_NEUT)]].values
        # Post-fix phenotype == signal exactly; pre-fix data is signal * 0.5.
        match_postfix = np.allclose(signal, recorded, atol=1e-4)
        match_prefix = np.allclose(signal * 0.5, recorded, atol=1e-4)
        tag = "post-fix" if match_postfix else "PRE-FIX (neut halved)" if match_prefix else "MISMATCH"
        print(f"  step {step:>7}: decoded bit-0 vs recorded neut_* -> {tag}")
        ok = ok and (match_postfix or match_prefix)
    return ok


def report(datadir):
    print(f"neutral baseline: bit-0 signal vs full 20-bit load "
          f"(equilibrium load = {MUT_EQUILIBRIUM:.4f})\n")
    for d in sorted(p for p in datadir.iterdir() if p.is_dir()):
        gdir = d / "snapshots" / "genotypes"
        snaps = sorted(gdir.glob("*.feather"), key=lambda p: int(p.stem))
        if not snaps:
            print(f"  {d.name:<28} no genotype snapshots "
                  f"(pull them: rsync .../snapshots/genotypes/)")
            continue
        gdf = pd.read_feather(snaps[-1])
        if gdf.empty:
            continue
        signal, load = decode(gdf)
        print(f"  {d.name:<28} step {snaps[-1].stem:>8}  "
              f"signal(bit0) {signal.mean():.4f}   load(20-bit) {load.mean():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=pathlib.Path,
                   default=pathlib.Path(os.environ.get("AEGIS_DATA",
                                        "~/aegis_data/ne_ma_ap_data")).expanduser(),
                   help="run output dir. Kept OUTSIDE Dropbox/git -- simulation output is\n                         GBs. Override with $AEGIS_DATA (e.g. the cluster path).")
    p.add_argument("--validate", metavar="PROBE_DIR", nargs="?", const="__local__",
                   help="self-check the column mapping against a run that has both "
                        "genotype and phenotype snapshots")
    args = p.parse_args()

    if args.validate is not None:
        probe = pathlib.Path(args.validate) if args.validate != "__local__" else None
        if probe is None:
            raise SystemExit("pass the probe run dir, e.g. --validate path/to/probe")
        print(f"validating column mapping on {probe}")
        raise SystemExit(0 if validate(probe) else 1)

    report(args.datadir)


if __name__ == "__main__":
    main()
