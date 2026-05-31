# AEGIS Manual

This is the operational manual for AEGIS (Aging of Evolving Genomes In
Silico) — a how-to reference for users running simulations and
analysing their outputs.

For the **scientific reference** — model design, ODD protocol,
biological motivation, validation experiments — see the published
paper:

> *AEGIS: Aging of Evolving Genomes In Silico.*
> Bagic M, Valenzano DR, et al. PLOS Computational Biology, 2026.
> [doi.org/10.1371/journal.pcbi.1014109](https://doi.org/10.1371/journal.pcbi.1014109)

This manual deliberately does **not** restate the model. Where a topic
is fully covered in the paper, the manual links there instead of
duplicating.

---

## Contents

1. [Installation](#installation)
2. [Running a simulation](#running-a-simulation)
3. [Trait phenotype ranges (`G_<trait>_lo` / `G_<trait>_hi`)](#trait-phenotype-ranges-g_trait_lo--g_trait_hi)
4. [Output formats](#output-formats)
5. [Output: FASTA vs VCF — when to use which](#output-fasta-vs-vcf--when-to-use-which)
6. [Lineage tracing and Muller plots](#lineage-tracing-and-muller-plots)
7. [Selection-coefficient experiments](#selection-coefficient-experiments)
8. [Introgression](#introgression)
9. [Post-processing scripts in `runs/`](#post-processing-scripts-in-runs)

(Sections grow over time as features land; this is a living document.)

---

## Installation

```bash
git clone https://github.com/valenzano-lab/aegis.git
cd aegis
git checkout v2
pip install -e ".[dev]"
```

Provides the `aegis` CLI and the `aegis_sim` Python package.

---

## Running a simulation

```bash
aegis sim -c path/to/config.yml         # fresh simulation
aegis sim -c path/to/config.yml -o      # overwrite existing output dir
aegis sim -c config.yml -r              # resume from checkpoint
aegis sim -c config.yml -r --extend N   # resume and extend to N steps
aegis gui                                # launch the Dash GUI
```

Configs are YAML. Outputs land in a directory named after the config
file: `path/to/config.yml` → `path/to/config/`.

---

## Trait phenotype ranges (`G_<trait>_lo` / `G_<trait>_hi`)

Every evolvable trait has a configurable phenotypic range — a floor and a ceiling. The composite architecture maps the interpreter's `[0, 1]` output onto the trait's `[lo, hi]` range:

```
phenotype = G_<trait>_lo + (G_<trait>_hi - G_<trait>_lo) * interpreter_output
```

So an individual whose locus bits give an interpreter output of 0.5 gets a phenotype at the **midpoint** of `[lo, hi]` — not 0.5.

This matters because small populations need a survival floor to be viable when `G_surv_initgeno=0.5` (the biologically motivated default — 50% random initial bits, both positive and negative selection observable from the outset). Without `lo > 0`, a fresh genome gives surv ~0.5 per age step → expected lifespan ~2 steps → guaranteed extinction.

Defaults (v2.3.2+):

| Trait | `lo` | `hi` | Notes |
|---|---|---|---|
| surv | **0.7** | 1.0 | floor at 0.7 means surv never collapses to 0; a 50%-genome individual has surv ~0.85 per age |
| repr | 0.0 | 0.5 | a 50%-genome fertile individual has ~0.25 chance to reproduce per step |
| neut | 0.0 | 1.0 | neutral by definition; no phenotype effect anyway |
| muta | 0.0 | 1.0 | scales the per-bit mutation rate |
| grow | 0.0 | 1.0 | grow trait is a stub today; effect TBD when growth is wired into the loop |

Set any of these per-config in YAML:

```yaml
G_surv_lo: 0.9          # high-survival regime (e.g. lab-protected populations)
G_repr_hi: 0.8          # higher fertility ceiling
```

Or interactively in the GUI's "genetics" accordion.

Note for users running long evolutionary simulations: a high `G_surv_lo` softens selection on surv — every individual gets at least `lo` survival regardless of how bad its surv-locus genome is. Use small `lo` (or zero) when you specifically want to study the death of bad-survival genotypes.

---

## Output formats

By default a sim produces a mix of CSV (per-step counters and rate
recorders), feather (per-snapshot population data), and pickle (full
Population objects). Several additional formats are opt-in via
config flags:

| Format        | Flag                | Purpose                                            |
|---------------|---------------------|----------------------------------------------------|
| Pickle        | `PICKLE_RATE` > 0   | Full Population object — full state, lossless     |
| FASTA         | `FASTA_RATE` > 0    | Per-individual DNA-like sequences (for Badread)   |
| VCF           | `VCF_RATE` > 0      | Per-bit biallelic SNP table (for PLINK/vcftools)  |
| Lineage CSV   | `LINEAGE_TRACING: true`, `LINEAGE_RATE` > 0 | Births + deaths log (for Muller plots, MRCA, ancestry trees) |
| Selection CSV | `ALLELE_INJECTION_STEP` > 0 | Allele-frequency trajectory at the injection locus (for fitting s) |

All opt-in outputs default to off and have no effect on the simulation
when disabled.

---

## Output: FASTA vs VCF — when to use which

Both formats encode the same underlying genome bits, but they're built
for different downstream tools and different scientific questions.

### FASTA — for the sequencing-simulation pipeline

- One FASTA record per individual; each record is the individual's
  full genome packed as a DNA-like sequence using a 4-letter encoding
  (00→A, 01→C, 10→G, 11→T) with a global uniform XOR mask to debias
  composition.
- Feeds directly into long-read simulators such as **Badread**.
  Workflow: `AEGIS → genome.fasta → Badread → simulated reads → aligner → variant caller → introgression detector`.
- Use when you want to model the *experimental pipeline including
  sequencing noise* — e.g. testing how well a Nanopore-based
  introgression detection method works under realistic read-error
  conditions.
- Not directly readable by population-genetics tools: you'd have to
  go through the full read → align → call → genotype detour just to
  recover the genotypes AEGIS already knows.
- Sidecar: `<name>.mapping.json` stores the XOR mask + shape so the
  round-trip back to original bits is lossless.

### VCF — for direct population-genetics analysis

- One row per genome bit (= one biallelic SNP), columns are
  individuals' phased genotypes (`0|1`, `1|0`, `0|0`, `1|1`).
- Directly readable by PLINK, vcftools, ADMIXTOOLS, scikit-allel,
  BCFtools — no conversion.
- Use for FST, ABBA-BABA, allele-frequency spectra, kinship matrices,
  PCA, ROH scans — anything that operates on genotypes rather than
  reads.
- Self-contained: header includes `##AEGIS_*` lines with the
  architecture metadata, so decoding back to the original
  multi-dimensional genome array doesn't require re-initializing
  AEGIS.

### How they complement each other in an introgression study

- **VCF gives you the ground truth** — what's actually in each
  individual's genome at each locus, computed directly. Combine with
  the `ancestry` array (when `INTROGRESSION_SEEDS > 0`) to know
  exactly which alleles came from which donor population.
- **FASTA + Badread gives you the experimental estimate** — what
  your downstream detection pipeline can *recover* under realistic
  sequencing noise.
- Comparing the two measures **pipeline accuracy under controlled
  conditions** — this is the core benchmarking value of AEGIS as a
  source of synthetic ground truth.

**Rule of thumb:** if your downstream tool expects reads, use FASTA.
If it expects genotypes, use VCF.

---

## Lineage tracing and Muller plots

(Section to be expanded.) See `runs/INTROGRESSION_FASTA_GUIDE.md` and
`runs/muller_plot.py` for the current state. Lineage tracing is
activated by `LINEAGE_TRACING: true` + `LINEAGE_RATE: > 0`; outputs
land in `<output>/lineage/{births,deaths}.csv`.

Only asexual reproduction is currently supported for lineage tracking;
with sexual reproduction the feature logs a warning and falls back
to no-op for offspring.

---

## Selection-coefficient experiments

(Section to be expanded.) Activate with `ALLELE_INJECTION_STEP > 0`
plus the trait/age/bit/allele/fraction parameters. The
SelectionRecorder writes `<output>/selection/selection.csv`; use
`runs/fit_s.py` to fit `log(p/(1-p)) ~ a + s·t` and obtain `s` with
95% CI.

This is an experimental intervention — a controlled allele knock-in
at a known locus — not a change to AEGIS's natural per-bit mutation
system (which is unchanged and still driven by the `muta` trait).

---

## Introgression

(Section to be expanded.) See `runs/INTROGRESSION_FASTA_GUIDE.md` for
the current end-to-end recipe. Key parameters: `INTROGRESSION_SOURCE`
(pickle path of pop A) + `INTROGRESSION_SEEDS` (number of pop-A
individuals to seed into the running pop B). The `ancestry` array on
each individual then tracks introgressed alleles locus-by-locus
through recombination and mutation.

---

## Post-processing scripts in `runs/`

| Script                       | Reads                                                | Produces                       |
|------------------------------|------------------------------------------------------|--------------------------------|
| `runs/fasta_roundtrip.py`    | a sim's pickle + FASTA                               | bit-equality + phenotype assertions, overlaid surv/repr plot |
| `runs/muller_plot.py <dir>`  | `<dir>/lineage/{births,deaths}.csv`                  | `<dir>/lineage/muller.png` (stacked-area founder dynamics) |
| `runs/fit_s.py <dir>`        | `<dir>/selection/selection.csv` + `final_config.yml` | s estimate with 95% CI, fit_s.png |

These are opt-in tools that read sim outputs after the fact — none of
them run during the simulation itself.
