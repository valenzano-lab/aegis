# Introgression + FASTA export — quick start

This guide shows how to use the two features added to AEGIS v2 for the introgression study:

- **Introgression machinery** (from v2.1): seed a second pre-evolved population into a running simulation and track introgressed alleles locus-by-locus.
- **FASTA export**: dump each individual's genome as a DNA-like FASTA record (with a sidecar JSON to allow exact decoding back to genome bits and phenotypes).

The FASTA output is designed to feed directly into a long-read simulator such as **Badread**.

---

## 1. Install

From a clean Python env:

```bash
git clone https://github.com/valenzano-lab/aegis.git
cd aegis
git checkout v2
pip install -e ".[dev]"
```

This gives you the `aegis` CLI and the `aegis_sim` Python package.

---

## 2. The two key parameters

Add these to any YAML config to turn FASTA output on:

```yaml
FASTA_RATE: 100         # dump FASTA every 100 steps (also at step 1 and final step). 0 = off.
FASTA_MASK_SEED: 0      # XOR mask seed for base-composition debiasing. Keep at 0 across all
                        # runs in an experiment so their FASTAs share an encoding coordinate system.
```

For introgression you need a pickled source population first, then a second sim that loads it:

```yaml
INTROGRESSION_SOURCE: /absolute/path/to/pop_a/pickles/<step>   # pickle of pop A
INTROGRESSION_SEEDS: 20                                         # how many pop-A individuals to seed
```

`INTROGRESSION_SEEDS: 0` disables introgression. When > 0, every individual carries an `ancestry` bit array (same shape as the genome): `True` = introgressed bit, `False` = native. Recombination and mutation track ancestry through generations.

---

## 3. Quick round-trip demo (verify the install)

There's a tiny example config and a script that round-trips the FASTA back to genomes + phenotypes:

```bash
aegis sim -c runs/fasta_test.yml -o     # writes runs/fasta_test/fasta/step1.{genome.fasta,mapping.json}
python runs/fasta_roundtrip.py          # encodes pickle → FASTA → decode → asserts bit + phenotype equality
```

Expected output:

```
OK bits: all 88960 bits identical across 139 individuals
OK phenotypes: max |diff| = 0.00e+00
```

A side-by-side phenotype plot is written to `runs/fasta_test/roundtrip.png`.

---

## 4. End-to-end introgression experiment recipe

The workflow is three separate `aegis sim` invocations:

### Step 1 — evolve pop A

Make `pop_a.yml`:

```yaml
RANDOM_SEED: 1
STEPS_PER_SIMULATION: 5000
AGE_LIMIT: 30
INITIAL_POPULATION_SIZE: 500
PICKLE_RATE: 0          # only at final step
FASTA_RATE: 0           # don't need FASTA from pop A alone
ENVDRIFT_RATE: 0
BITS_PER_LOCUS: 8
# Strong contrast at the trait of interest:
G_neut_initgeno: 1.0    # neutral-locus arm: pop A starts all 1s at neutral loci
# (or, for the survival arm:)
# G_surv_initgeno: 1.0
# G_repr_initgeno: 1.0
```

Run:

```bash
aegis sim -c pop_a.yml
# produces pop_a/pickles/5000
```

### Step 2 — evolve pop B (opposite extreme)

`pop_b.yml`: same as `pop_a.yml` but `RANDOM_SEED: 2` and the opposite extreme:

```yaml
G_neut_initgeno: 0.0    # pop B starts all 0s at neutral loci
```

```bash
aegis sim -c pop_b.yml
```

### Step 3 — introgression sim

`introgress.yml`: continue from pop B, seed N individuals from pop A:

```yaml
RANDOM_SEED: 3
STEPS_PER_SIMULATION: 1000
AGE_LIMIT: 30
INITIAL_POPULATION_SIZE: 500
PICKLE_RATE: 500
FASTA_RATE: 500
FASTA_MASK_SEED: 0      # same mask as anything else you compare against
ENVDRIFT_RATE: 0
BITS_PER_LOCUS: 8

INTROGRESSION_SOURCE: /full/path/to/pop_a/pickles/5000
INTROGRESSION_SEEDS: 20
```

Start from pop B's final pickle:

```bash
aegis sim -c introgress.yml --pickle pop_b/pickles/5000
```

You now have:

- `introgress/fasta/step{N}.genome.fasta` — XOR-masked FASTA, one record per living individual
- `introgress/fasta/step{N}.mapping.json` — the ledger (mask + shape) needed to decode
- `introgress/pickles/{N}` — full Population pickle (includes `ancestry` array as ground truth)

### Step 4 — run Badread on the FASTA

```bash
badread simulate \
  --reference introgress/fasta/step1000.genome.fasta \
  --quantity 30x \
  --length 5000,2000 \
  --error_model nanopore2023 --qscore_model nanopore2023 \
  --identity 95,99,4 --glitches 1000,25,25 \
  --junk_reads 1 --random_reads 1 \
  > introgress/reads.fastq
```

### Step 5 — score detected introgression against ground truth

Whatever pipeline you use to call introgressed regions from the reads, the *ground truth* labels for every base are in `population.ancestry` from the pickle. To get them:

```python
import pickle
from aegis_sim.dataclasses.population import Population

pop = Population.load_pickle_from("introgress/pickles/1000")
# pop.ancestry has the same shape as pop.genomes.array
# True at a bit position = that bit came from pop A
```

---

## 5. Decode a FASTA back to genomes/phenotypes programmatically

```python
import pathlib, numpy as np
from aegis_sim.utilities.fasta import decode_fasta_to_genomes
from aegis_sim.dataclasses.genomes import Genomes

decoded_array, record_ids = decode_fasta_to_genomes(
    pathlib.Path("introgress/fasta/step1000.genome.fasta"),
    pathlib.Path("introgress/fasta/step1000.mapping.json"),
)
# decoded_array has the same shape and dtype (bool) as population.genomes.array
genomes = Genomes(decoded_array)
```

To derive phenotypes from a decoded genome, you need an initialized `architect` (it needs the simulation parameters). See `runs/fasta_roundtrip.py` for the `init_without_recording()` helper that bootstraps the architect without re-running the sim.

---

## 6. About the FASTA encoding

- **4-letter packing**: every 2 genome bits become 1 base. `00→A, 01→C, 10→G, 11→T`.
- **Global XOR mask**: applied to all bits before packing. This removes the composition bias that would otherwise appear (AEGIS-fit individuals have many `1`s, which without masking would produce T-heavy sequences and break Badread's error model + downstream aligners). The mask is uniform-random, so the encoded composition is ~25% per base regardless of the underlying selection pattern.
- **Mapping ledger** (`mapping.json`): `xor_mask_hex`, original genome `shape`, `bits_per_individual`, `padding_bits`, `encoding` name, and the bit-pair → base table. Everything needed to invert the encoding losslessly.
- **Ledger consistency across runs**: `FASTA_MASK_SEED` is independent of `RANDOM_SEED`. As long as `FASTA_MASK_SEED` and the genome architecture (`BITS_PER_LOCUS`, `AGE_LIMIT`, ploidy, traits) match across runs, the mapping ledger is byte-identical. Different replicates of the same experiment can be compared directly.
- **Hamming distance is preserved**: XOR is linear, so within-pop pairs stay close in sequence space and cross-pop pairs stay far. Nothing about the introgression signal is destroyed by the encoding.

---

## 7. Known limitations / TODOs

- **Ancestry sidecar FASTA** (e.g. `step{N}.ancestry.fasta` with `A`=native, `T`=introgressed) is **not yet implemented**. For now the ground-truth ancestry lives only in `population.ancestry` on the pickle. If you need it as a FASTA-aligned track for the analysis pipeline, ask Dario.
- **Per-individual phenotype TSV sidecar** is also not yet emitted. Decode the FASTA and run the architect to recompute phenotypes (see Section 5).
- **Position-to-trait mapping** (which base in the FASTA corresponds to which trait × age × bit) is not in `mapping.json` yet — only the structural shape is. The decoder reconstructs the full multi-dimensional genome array, so once decoded you can index by `[individual, ploidy, locus, bit_within_locus]` directly.

If any of these blocks your analysis, let Dario know and they can be added quickly.

---

## 8. Files of interest in the repo

- `src/aegis_sim/utilities/fasta.py` — `encode_population_to_fasta()` and `decode_fasta_to_genomes()`.
- `src/aegis_sim/recording/fastarecorder.py` — the recorder that runs during simulation when `FASTA_RATE > 0`.
- `src/aegis_sim/__init__.py` (`_seed_introgression`) — the introgression seeding logic.
- `src/aegis_sim/recording/ancestryrecorder.py` — writes mean introgression fraction per locus over time (`ancestry.csv` in the output dir).
- `runs/fasta_test.yml` — minimal config that exercises the FASTA path.
- `runs/fasta_roundtrip.py` — reference script for decoding a FASTA back to phenotypes.
