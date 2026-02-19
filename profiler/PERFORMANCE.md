# AEGIS Performance Analysis

Profiled February 2026. Results from macOS, Python 3.12, NumPy + Numba.

## Summary

| Config | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| Default (composite, BPL=1, pop ~550) | 4.93ms/step | 0.92ms/step | 5.3x |
| Hard nemaap (modifying, sexual, 2000 loci, R=25600, pop ~6000) | 16.05ms/step (8.9h for 2M steps) | 6.3ms/step (3.5h for 2M steps) | 2.5x |

## Optimizations Applied

### 1. Recombination (`recombination_via_pairs_numba`)

Replaced slice-copy swap pattern with difference-array + prefix-sum approach.
O(n_chiasmata × n_sites) → O(n_chiasmata + n_sites).

File: `src/aegis_sim/submodels/reproduction/recombination.py`

| BITS_PER_LOCUS | Original | Optimized | Speedup |
|----------------|----------|-----------|---------|
| 1              | 0.24ms   | 0.06ms    | 4-5x    |
| 8              | 8.95ms   | 0.40ms    | 22-28x  |

### 2. Buffered I/O for per-step recorders

`PopsizeRecorder` and `ResourcesRecorder` buffer writes in memory, flushing every
100 entries or at checkpoint time. Eliminates 5 file open/close cycles per step.

Files: `popsizerecorder.py`, `resourcerecorder.py`, `checkpointrecorder.py`, `__init__.py`

Recording overhead: ~86ms → ~4ms per 200 steps (21x).

### 3. Diploid-to-haploid conversion (`ploider.diploid_to_haploid`)

Replaced `logical_or` + `astype(float64)` + `logical_xor` + scattered mask write
with a parallel numba kernel. Single pass, float32 output, `prange` over individuals.

File: `src/aegis_sim/submodels/genetics/ploider.py`

46.9ms → 2.4ms per call at 6000 individuals (19.5x).

### 4. Phenodiff kernel (`apply_phenolist_numba`)

Eliminated 72MB temporary array (`vec_states = vectors[:, vec_indices]`). Kernel now
indexes directly into `vectors` with `prange` over individuals. Phenolist index arrays
cached on the GPM object.

File: `src/aegis_sim/submodels/genetics/modifying/gpm.py`

108.9ms → 3.0ms per call at 6000 individuals (36x).

### 5. Genomes init skip unnecessary astype

`Genomes.__init__` checks dtype before calling `.astype(np.bool_)`, skipping the copy
when input is already bool.

File: `src/aegis_sim/dataclasses/genomes.py`

~0.3-0.5ms/step saved. Zero risk.

### 6. Pairing gamete assembly (`pairing`)

Replaced 4 intermediate array allocations (two `genomes.get` + two gamete selections +
copy into children) with a single numba `prange` kernel that reads parent chromatids
directly and writes children in one pass.

File: `src/aegis_sim/submodels/reproduction/pairing.py`

3.86ms → 1.19ms per call at 3000 pairs (3.2x).

### 7. Packed bit genome storage (`Genomes`)

Replaced `np.bool_` genome storage (1 byte per bit) with `np.uint8` packed storage
(8 bits per byte). The `Genomes` class stores a packed array internally; `keep()`,
`add()`, `__getitem__()` operate on packed data directly. `get()`, `unpack()`,
`flatten()`, `get_array()` transparently unpack to bool for downstream modules.

Files: `src/aegis_sim/dataclasses/genomes.py`, `src/aegis_sim/dataclasses/legacy_genomes.py`

8x memory reduction for genome arrays. At 6000 individuals: 24MB → 3MB.
Mortality (`keep`) and hatching (`add`) ~40-70% faster due to smaller copies.

The original `Genomes` class is preserved as `LegacyGenomes` for permanent
reference testing. Non-divisible-by-8 genome sizes are handled with padding
(warning logged).

## Hard Config Final Breakdown (6.3ms/step)

| Phase | Avg (ms) | % | Dominant cost |
|-------|----------|---|---------------|
| Reproduction | ~4.2 | ~68% | recombination, genomes.get (unpack) |
| Hatching | ~1.0 | ~16% | ploider + phenodiff |
| Mortalities | ~0.7 | ~11% | genomes.keep (packed, fast) |
| Aging | ~0.2 | ~3% | genomes.keep (packed, fast) |
| Recording | ~0.1 | ~1% | buffered, negligible |

## Population Scaling (packed bit genomes, modifying architecture, 2000 loci)

| Population | Genome memory | ms/step | 2M steps |
|-----------|--------------|---------|----------|
| 1,000 | 0.8 MB | 2.5ms | 1.4h |
| 5,000 | 2.5 MB | 5.6ms | 3.1h |
| 10,000 | 4.0 MB | 7.0ms | 3.9h |
| 25,000 | 8.6 MB | 11.6ms | 6.4h |
| 50,000 | 16.2 MB | 18.8ms | 10.5h |

Scaling is roughly linear with population size. Without packed storage, 50K
individuals would require ~130MB for genomes alone.

## What's Left (diminishing returns)

The remaining time is dominated by the `get()` unpack cost in reproduction:

| Operation | What it does | Why it's hard to optimize further |
|-----------|-------------|-----------------------------------|
| `genomes.get` + unpack | Extracts parental genomes as bool for recombination/mutation | Would need recombination and mutation to work directly on packed bytes |
| `genomes.keep` | Filters packed array after each kill | Already fast on packed data |
| `genomes.add` | Concatenates packed arrays for offspring | Already fast on packed data |

### Future: packed-native mutation and recombination

Tasks 4 and 5 from the packed-bit-genomes spec describe adapting mutation (`by_index`
with XOR masks on packed bytes) and recombination (byte-level swaps with bit masking
at crossover boundaries) to work directly on packed data. This would eliminate the
`get()` unpack cost in reproduction (~4ms/step), but requires careful bit-level
programming and thorough testing. See `.kiro/specs/packed-bit-genomes/design.md`
for the detailed design.

### Why mortality batching was not pursued

Mortality sources have sequential dependencies — infection mutates population state,
predation uses Verhulst dynamics based on current prey count, starvation depends on
current population size for resource demand. `MORTALITY_ORDER` is configurable because
ordering matters biologically.

### Next-level optimizations (architectural changes)

#### Pre-allocated population buffer

Instead of `np.concatenate` to add offspring and fancy-index to remove dead, maintain
a fixed-capacity buffer with a live-index list. Adding offspring writes into empty
slots. Killing marks slots as dead. Compaction happens once per step or less.

This eliminates repeated full-array copies but doesn't reduce the per-element cost.
More invasive because it changes how every module accesses population data
(indirection through live indices or views).

Estimated impact: eliminates ~1-2ms/step of copy overhead. More impactful at very
large populations where allocation/copy dominates.

## Known Technical Debt

### Dual RNG streams

The codebase uses both the legacy global RNG (`np.random.*`) and the new-style
Generator (`variables.rng`). Both are seeded from `RANDOM_SEED` so simulations are
reproducible, but the split is unprincipled — there's no clear rule for which modules
use which. See TODO in `src/aegis_sim/variables.py`.

## Profiling Scripts

- `profile_sim.py` — cProfile of the full simulation loop
- `profile_breakdown.py` — Wall-clock timing of each phase within `run_step()`

```
python profiler/profile_sim.py
python profiler/profile_breakdown.py
```
