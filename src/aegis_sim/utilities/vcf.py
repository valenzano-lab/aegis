"""VCF export/import for AEGIS genomes (composite architecture).

Each bit position is treated as a biallelic SNP (REF=A, ALT=T by convention).
Each individual is one sample column with phased genotype CHROMATID0|CHROMATID1.
Rows are ordered by logical (trait × age × bit-within-locus) position so that
adjacent rows are biologically adjacent loci — useful for downstream tools that
look at local linkage or sliding-window stats.

The VCF header records the `locus_permutation` and architecture metadata so the
file is self-contained: `decode_vcf_to_genomes()` can reconstruct the exact
multi-dimensional genome array without re-initializing AEGIS.

Only the composite architecture is supported here; modifying-architecture export
would need a different structural mapping.
"""

import pathlib
from typing import Tuple, List, Dict

import numpy as np

REF_BASE = "A"
ALT_BASE = "T"


def encode_population_to_vcf(
    population,
    output_dir: pathlib.Path,
    name: str = "dump",
) -> pathlib.Path:
    """Write a VCF with one row per genome bit and one sample column per individual.

    Pulls architecture metadata (locus_permutation, trait layout) from
    `aegis_sim.submodels.architect`, so the architect must be initialized.

    Returns the path to the written VCF.
    """
    from aegis_sim import submodels, parameterization

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture = submodels.architect.architecture
    if not hasattr(architecture, "locus_permutation"):
        raise RuntimeError("VCF export only supports the composite architecture")

    genome_array = population.genomes.array
    n_individuals, ploidy, n_loci, bits_per_locus = genome_array.shape
    if ploidy != 2:
        raise ValueError(f"VCF export expects diploid genomes (ploidy=2), got ploidy={ploidy}")

    locus_permutation = np.asarray(architecture.locus_permutation, dtype=int)
    traits = parameterization.traits

    sample_names: List[str] = []
    birthdays = getattr(population, "birthdays", None)
    ages = getattr(population, "ages", None)
    for i in range(n_individuals):
        b = int(birthdays[i]) if birthdays is not None else -1
        a = int(ages[i]) if ages is not None else -1
        sample_names.append(f"ind_{i}_b{b}_a{a}")

    vcf_path = output_dir / f"{name}.vcf"
    with open(vcf_path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##source=AEGIS\n")
        fh.write(f"##contig=<ID=aegis_genome,length={n_loci * bits_per_locus}>\n")
        fh.write('##INFO=<ID=TRAIT,Number=1,Type=String,Description="Trait name">\n')
        fh.write('##INFO=<ID=AGE,Number=1,Type=Integer,Description="Age (0-indexed) for age-specific traits, -1 otherwise">\n')
        fh.write('##INFO=<ID=BIT,Number=1,Type=Integer,Description="Bit position within the BITS_PER_LOCUS locus (0-indexed)">\n')
        fh.write('##INFO=<ID=PHYS_LOCUS,Number=1,Type=Integer,Description="Physical locus index in storage">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype: chromatid0|chromatid1">\n')
        fh.write(f"##AEGIS_BITS_PER_LOCUS={bits_per_locus}\n")
        fh.write(f"##AEGIS_N_LOCI={n_loci}\n")
        fh.write(f"##AEGIS_PLOIDY={ploidy}\n")
        fh.write(f"##AEGIS_N_INDIVIDUALS={n_individuals}\n")
        fh.write(f"##AEGIS_LOCUS_PERMUTATION={','.join(str(x) for x in locus_permutation.tolist())}\n")
        fh.write("#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + sample_names) + "\n")

        for logical_locus in range(n_loci):
            physical_locus = int(locus_permutation[logical_locus])
            trait_name, age = _logical_to_trait_age(logical_locus, traits)
            for bit_in_locus in range(bits_per_locus):
                pos = logical_locus * bits_per_locus + bit_in_locus + 1  # 1-indexed
                variant_id = f"{trait_name}_a{age}_b{bit_in_locus}" if age >= 0 else f"{trait_name}_b{bit_in_locus}"
                info = f"TRAIT={trait_name};AGE={age};BIT={bit_in_locus};PHYS_LOCUS={physical_locus}"

                # Pull bits for all individuals at this physical position
                chr0 = genome_array[:, 0, physical_locus, bit_in_locus].astype(np.int8)
                chr1 = genome_array[:, 1, physical_locus, bit_in_locus].astype(np.int8)
                # Phased genotype strings
                gts = [f"{int(chr0[i])}|{int(chr1[i])}" for i in range(n_individuals)]

                row = ["aegis_genome", str(pos), variant_id, REF_BASE, ALT_BASE, ".", "PASS", info, "GT"] + gts
                fh.write("\t".join(row) + "\n")

    return vcf_path


def _logical_to_trait_age(logical_locus: int, traits) -> Tuple[str, int]:
    """Map a logical locus index to (trait_name, age). age=-1 for non-age-specific."""
    for trait in traits.values():
        if trait.length == 0:
            continue
        if trait.start <= logical_locus < trait.end:
            age = logical_locus - trait.start if trait.agespecific is True else -1
            return trait.name, age
    raise IndexError(f"logical_locus {logical_locus} is outside any trait range")


def decode_vcf_to_genomes(vcf_path: pathlib.Path) -> Tuple[np.ndarray, List[str]]:
    """Parse an AEGIS-emitted VCF back to (genome_array, sample_names).

    The VCF header contains the metadata needed to reconstruct the original
    (n_individuals, ploidy, n_loci, bits_per_locus) array — no AEGIS init required.
    """
    vcf_path = pathlib.Path(vcf_path)

    meta: Dict[str, str] = {}
    sample_names: List[str] = []
    data_rows: List[Tuple[int, int, int, List[Tuple[int, int]]]] = []  # phys_locus, bit, _, genotypes

    with open(vcf_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("##"):
                if "=" in line:
                    key, _, value = line[2:].partition("=")
                    meta[key] = value
                continue
            if line.startswith("#CHROM"):
                fields = line.lstrip("#").split("\t")
                sample_names = fields[9:]
                continue
            if not line:
                continue
            fields = line.split("\t")
            info = dict(kv.split("=") for kv in fields[7].split(";"))
            phys_locus = int(info["PHYS_LOCUS"])
            bit_in_locus = int(info["BIT"])
            gt_strings = fields[9:]
            genotypes = []
            for gt in gt_strings:
                a, b = gt.split("|")
                genotypes.append((int(a), int(b)))
            data_rows.append((phys_locus, bit_in_locus, 0, genotypes))

    n_individuals = int(meta["AEGIS_N_INDIVIDUALS"])
    ploidy = int(meta["AEGIS_PLOIDY"])
    n_loci = int(meta["AEGIS_N_LOCI"])
    bits_per_locus = int(meta["AEGIS_BITS_PER_LOCUS"])
    assert len(sample_names) == n_individuals
    assert len(data_rows) == n_loci * bits_per_locus

    genome_array = np.zeros((n_individuals, ploidy, n_loci, bits_per_locus), dtype=np.bool_)
    for phys_locus, bit_in_locus, _, genotypes in data_rows:
        for ind_idx, (a, b) in enumerate(genotypes):
            genome_array[ind_idx, 0, phys_locus, bit_in_locus] = bool(a)
            genome_array[ind_idx, 1, phys_locus, bit_in_locus] = bool(b)

    return genome_array, sample_names
