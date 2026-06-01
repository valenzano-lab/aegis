"""FASTA-coordinate multi-sample gVCF export for AEGIS populations.

Designed as a drop-in replacement for Clair3-emitted gVCFs in pipelines
that joint-genotype via GLnexus and then run population-genetics tests
like ABBA-BABA. Output is one gVCF per Population, each individual is a
sample column, each row is a FASTA base position with the same coordinate
system as the FASTA + Badread reads.

Per position:
  REF = consensus (modal) base across all 2*N alleles
  ALT = sorted list of observed non-REF bases + <NON_REF>
  GT  = allele indices into [REF, *ALTs] for each diploid individual
  GQ  = 99 (synthetic; AEGIS has no read uncertainty)
  DP  = 30 (synthetic)
  AD  = per-allele depth, fixed at 15 for the called genotype's alleles

Long runs of all-REF positions are compressed into reference blocks
(single record with END=<last_pos_of_block>, ALT=<NON_REF>) — standard
gVCF convention, expected by GLnexus.

Only the composite architecture is supported. Modifying architecture
output would need a separate emitter (different locus layout).
"""

import io
import pathlib
from typing import List, Tuple

import numpy as np

BASES = np.array(["A", "C", "G", "T"])
NON_REF = "<NON_REF>"


def encode_population_to_gvcf(
    population,
    output_dir: pathlib.Path,
    name: str = "dump",
) -> pathlib.Path:
    """Write a FASTA-coordinate multi-sample gVCF for the given Population.

    Coordinates and bases match what `encode_population_to_fasta` produces:
    same XOR mask seed, same 4-letter packing, same physical/logical ordering.
    Returns the path to the written gVCF.
    """
    from aegis_sim import submodels
    from aegis_sim.parameterization import parametermanager

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture = submodels.architect.architecture
    if not hasattr(architecture, "locus_permutation"):
        raise RuntimeError("gVCF export only supports the composite architecture")

    base_indices_phys = _decode_to_base_indices(population, parametermanager)
    # base_indices_phys shape: (n_individuals, ploidy, n_base_positions) in PHYSICAL order
    # We emit rows in LOGICAL order so adjacent rows are biologically adjacent loci.
    base_indices = _physical_to_logical_bases(
        base_indices_phys,
        locus_permutation=np.asarray(architecture.locus_permutation, dtype=int),
        bits_per_locus=architecture.BITS_PER_LOCUS,
    )

    n_individuals, ploidy, n_positions = base_indices.shape
    if ploidy != 2:
        raise ValueError(f"gVCF export expects diploid (ploidy=2), got ploidy={ploidy}")

    # Per-position consensus REF + ALT list
    refs, alts_per_pos = _consensus_ref_and_alts(base_indices)

    sample_names = _sample_names(population, n_individuals)

    gvcf_path = output_dir / f"{name}.gvcf"
    with open(gvcf_path, "w") as fh:
        _write_header(fh, n_positions, sample_names)
        _write_body(fh, base_indices, refs, alts_per_pos, sample_names)

    return gvcf_path


# ----- decode + per-position aggregation -------------------------------------

def _decode_to_base_indices(population, parametermanager) -> np.ndarray:
    """Replicate the FASTA encoding pipeline up to the point where we know
    each individual's bases. Returns int8 array of shape (n_ind, 2, n_bases)
    in PHYSICAL locus order, with values in {0,1,2,3} for {A,C,G,T}."""
    genome_array = population.genomes.array  # (n, 2, n_loci, bits_per_locus) bool
    n_individuals, ploidy, n_loci, bits_per_locus = genome_array.shape

    flat = genome_array.reshape(n_individuals, ploidy, -1).astype(np.uint8)
    total_bits = flat.shape[-1]
    pad = total_bits % 2
    if pad:
        flat = np.concatenate(
            [flat, np.zeros((n_individuals, ploidy, 1), dtype=np.uint8)], axis=-1
        )
    padded_length = total_bits + pad

    mask_seed = int(getattr(parametermanager.parameters, "FASTA_MASK_SEED", 0))
    mask = np.random.default_rng(mask_seed).integers(0, 2, size=padded_length, dtype=np.uint8)
    masked = flat ^ mask  # broadcasts over (n_ind, ploidy)

    pairs = masked.reshape(n_individuals, ploidy, padded_length // 2, 2)
    base_indices = (pairs[..., 0] * 2 + pairs[..., 1]).astype(np.int8)
    return base_indices  # (n_ind, 2, n_bases)


def _physical_to_logical_bases(
    base_indices_phys: np.ndarray,
    locus_permutation: np.ndarray,
    bits_per_locus: int,
) -> np.ndarray:
    """Reorder bases from physical-locus order to logical-locus order.

    Each logical locus has BITS_PER_LOCUS bits → bits_per_locus/2 bases.
    """
    n_individuals, ploidy, n_bases = base_indices_phys.shape
    bases_per_locus = bits_per_locus // 2
    if bases_per_locus * bits_per_locus // 2 != bases_per_locus:
        # Handles odd bits_per_locus by leaving the last base in its physical slot;
        # for the standard 8-bits-per-locus case, bases_per_locus=4 and this is exact.
        pass
    n_loci = n_bases // bases_per_locus
    by_locus = base_indices_phys[:, :, : n_loci * bases_per_locus].reshape(
        n_individuals, ploidy, n_loci, bases_per_locus
    )
    by_locus = by_locus[:, :, locus_permutation, :]
    out = by_locus.reshape(n_individuals, ploidy, n_loci * bases_per_locus)
    # Append any leftover trailing bases (rare; only if bits_per_locus is odd)
    if n_bases > n_loci * bases_per_locus:
        out = np.concatenate(
            [out, base_indices_phys[:, :, n_loci * bases_per_locus :]], axis=-1
        )
    return out


def _consensus_ref_and_alts(
    base_indices: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Per-position consensus REF + sorted list of observed non-REF ALTs.

    base_indices is (n_ind, 2, n_bases). Returns:
      refs:          int8 (n_bases,) — modal base index per position
      alts_per_pos:  list of lists; alts_per_pos[p] is the sorted ALT indices
    """
    n_individuals, ploidy, n_bases = base_indices.shape
    flat = base_indices.transpose(2, 0, 1).reshape(n_bases, -1)  # (n_bases, n_ind*ploidy)

    refs = np.zeros(n_bases, dtype=np.int8)
    alts_per_pos: List[List[int]] = []
    for p in range(n_bases):
        counts = np.bincount(flat[p], minlength=4)
        ref = int(np.argmax(counts))
        refs[p] = ref
        observed = sorted(int(b) for b in np.unique(flat[p]) if int(b) != ref)
        alts_per_pos.append(observed)
    return refs, alts_per_pos


# ----- writers ---------------------------------------------------------------

def _sample_names(population, n_individuals: int) -> List[str]:
    birthdays = getattr(population, "birthdays", None)
    ages = getattr(population, "ages", None)
    names = []
    for i in range(n_individuals):
        b = int(birthdays[i]) if birthdays is not None else -1
        a = int(ages[i]) if ages is not None else -1
        names.append(f"ind_{i}_b{b}_a{a}")
    return names


def _write_header(fh, n_positions: int, sample_names: List[str]) -> None:
    fh.write("##fileformat=VCFv4.2\n")
    fh.write("##source=AEGIS\n")
    fh.write(f"##contig=<ID=aegis_genome,length={n_positions}>\n")
    fh.write('##ALT=<ID=NON_REF,Description="Represents any possible alternative allele not already in ALT">\n')
    fh.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the reference block">\n')
    fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    fh.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
    fh.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n')
    fh.write('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele Depth">\n')
    fh.write("##GVCFBlock=AEGIS-synthetic\n")
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + sample_names
    fh.write("#" + "\t".join(columns) + "\n")


def _write_body(
    fh,
    base_indices: np.ndarray,
    refs: np.ndarray,
    alts_per_pos: List[List[int]],
    sample_names: List[str],
) -> None:
    """Walk positions emitting either a variant record or accumulating a
    reference block. Variant: any position with a non-empty ALT list.
    Reference block: a run of consecutive positions all with empty ALTs and
    same REF — emitted as one record with END=<last>.
    """
    n_individuals, ploidy, n_positions = base_indices.shape

    p = 0
    while p < n_positions:
        if not alts_per_pos[p]:
            # Start of (or continuation of) a reference block.
            block_ref = int(refs[p])
            block_start = p
            while p < n_positions and not alts_per_pos[p] and int(refs[p]) == block_ref:
                p += 1
            block_end = p  # exclusive
            _emit_ref_block(
                fh, start=block_start, end=block_end, ref=block_ref, n_individuals=n_individuals
            )
        else:
            _emit_variant(
                fh,
                pos=p,
                ref=int(refs[p]),
                alts=alts_per_pos[p],
                base_indices=base_indices,
            )
            p += 1


def _emit_ref_block(fh, start: int, end: int, ref: int, n_individuals: int) -> None:
    """One record covering positions [start, end), all individuals 0/0:99:30:30,0."""
    pos = start + 1  # 1-indexed POS
    end_pos = end  # END is 1-indexed inclusive == 0-indexed exclusive end
    fields = [
        "aegis_genome",
        str(pos),
        ".",
        BASES[ref],
        NON_REF,
        ".",
        ".",
        f"END={end_pos}",
        "GT:GQ:DP:AD",
    ]
    sample_strs = ["0/0:99:30:30,0"] * n_individuals
    fh.write("\t".join(fields + sample_strs) + "\n")


def _emit_variant(
    fh,
    pos: int,
    ref: int,
    alts: List[int],
    base_indices: np.ndarray,
) -> None:
    """One variant record at position pos. Alleles = [REF, *alts, <NON_REF>]."""
    n_individuals = base_indices.shape[0]
    all_alleles = [ref, *alts]  # index 0..len(alts) maps to actual base indices
    alt_strs = [BASES[a] for a in alts] + [NON_REF]
    ref_str = BASES[ref]

    base_to_allele_idx = {b: i for i, b in enumerate(all_alleles)}
    n_distinct = len(all_alleles) + 1  # +1 for <NON_REF>

    sample_strs: List[str] = []
    for i in range(n_individuals):
        a0 = int(base_indices[i, 0, pos])
        a1 = int(base_indices[i, 1, pos])
        # All observed alleles are in all_alleles by construction
        gt0 = base_to_allele_idx[a0]
        gt1 = base_to_allele_idx[a1]
        # AD: one entry per allele in REF + ALTs + NON_REF (1+len(alts)+1 = n_distinct)
        ad = [0] * n_distinct
        # Synthetic 15-read AD for each chromatid's allele
        ad[gt0] += 15
        ad[gt1] += 15
        ad_str = ",".join(str(x) for x in ad)
        sample_strs.append(f"{gt0}/{gt1}:99:30:{ad_str}")

    fields = [
        "aegis_genome",
        str(pos + 1),
        ".",
        ref_str,
        ",".join(alt_strs),
        ".",
        "PASS",
        ".",
        "GT:GQ:DP:AD",
    ]
    fh.write("\t".join(fields + sample_strs) + "\n")
