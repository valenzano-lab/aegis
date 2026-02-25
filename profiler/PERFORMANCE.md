# AEGIS Performance Analysis

Profiled February 2026. Results from macOS, Python 3.12, NumPy + Numba.

## Optimizations Applied

### 1. Recombination (`recombination_via_pairs_numba`)

The original numba function swapped genome slices with `.copy()` for each chiasma,
scaling as O(n_chiasmata × n_sites). Replaced with a difference-array + prefix-sum
approach that computes the net swap state per site in O(n_chiasmata + n_sites).

File: `src/aegis_sim/submodels/reproduction/recombination.py`

Microbenchmark (500 offspring, 250 loci):

| BITS_PER_LOCUS | Original | Optimized | Speedup |
|----------------|----------|-----------|---------|
| 1              | 0.24ms   | 0.06ms    | 4-5x    |
| 8              | 8.95ms   | 0.40ms    | 22-28x  |

### 2. Buffered I/O for per-step recorders

`PopsizeRecorder` and `ResourcesRecorder` were opening/closing files 5 times per step.
Now they buffer writes in memory and flush every 100 entries or at checkpoint time.

Files:
- `src/aegis_sim/recording/popsizerecorder.py`
- `src/aegis_sim/recording/resourcerecorder.py`
- `src/aegis_sim/recording/checkpointrecorder.py` (flushes buffers before saving)
- `src/aegis_sim/__init__.py` (flushes buffers at end of simulation)

Recording overhead dropped from ~86ms to ~4ms per 200 steps (~21x).

### 3. Diploid-to-haploid conversion (`ploider.diploid_to_haploid`)

The original code created three intermediate arrays (`logical_or` → `astype(float64)`
→ `logical_xor` → scattered mask write). Replaced with a parallel numba kernel that
reads both chromatids once and writes float32 output directly, using `prange` over
individuals.

File: `src/aegis_sim/submodels/genetics/ploider.py`

| Version | Time (6000 ind, 2000 loci) | Speedup |
|---------|---------------------------|---------|
| Original (or + xor + scatter, f64) | 46.9ms | 1x |
| np.where (f64) | 12.0ms | 3.9x |
| numba prange (f32) | 2.4ms | 19.5x |

### 4. Phenodiff kernel (`apply_phenolist_numba`)

The original kernel pre-gathered `vec_states = vectors[:, vec_indices]` (allocating a
72MB temporary for 6000 individuals × 3000 phenolist entries), then ran a single-threaded
double loop. Replaced with a parallel kernel that indexes directly into `vectors`,
eliminating the temporary and parallelizing over individuals with `prange`. Phenolist
index arrays are now cached on the GPM object (resolved once, reused every call).

File: `src/aegis_sim/submodels/genetics/modifying/gpm.py`

| Version | Time (6000 ind, 3000 phenolist) | Speedup |
|---------|--------------------------------|---------|
| Original (gather + sequential) | 108.9ms | 1x |
| Parallel, no gather | 3.0ms | 36x |

### 5. Genomes init skip unnecessary astype

`Genomes.__init__` always called `.astype(np.bool_)` even when the input was already
bool, creating an unnecessary copy. Now checks dtype first.

File: `src/aegis_sim/dataclasses/genomes.py`

Minor saving (~0.3-0.5ms/step), but zero-risk and eliminates wasteful allocations.

## End-to-End Results

### Default config (composite, BPL=1, pop ~550)

Per-step time: 4.93ms → 0.92ms (5.3x overall speedup)

### Hard config: nemaap (modifying, sexual, 2000 loci, R=25600, pop ~6000)

| Metric | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| Per step | 16.05ms | 8.4ms | 1.9x |
| 2M step runtime | 8.9 hours | 4.7 hours | 1.9x |

Phase breakdown after all optimizations:

| Phase | Avg (ms) | % |
|-------|----------|---|
| Reproduction | 4.37 | 51.9% |
| Hatching | 1.65 | 19.6% |
| Mortalities | 1.40 | 16.6% |
| Aging | 0.90 | 10.7% |
| Recording | 0.09 | 1.1% |

## Remaining Opportunities (diminishing returns)

The remaining time is spread across many small memory-bandwidth-bound operations.
Further optimization would require either significant refactors or architectural changes.

| Opportunity | Est. saving | Effort | Risk | Notes |
|---|---|---|---|---|
| `genomes.add` pre-allocated buffer | ~1ms/step | High | Medium | Refactor Genomes + Population to manage capacity/length |
| `pairing` fancy indexing | ~0.5ms/step | Medium | Low | Fuse two `genomes.get` + gamete selection into one kernel |
| `genomes.keep` boolean mask | ~0.3ms/step | Low | Low | Minor — callers already use bool masks in some cases |
| `phenotypes.extract` scratch buffer | ~0.2ms/step | Low | Low | Reuse pre-allocated array instead of `np.zeros` each call |

### Why mortality batching was not pursued

Mortality sources have sequential dependencies — infection mutates population state,
predation uses Verhulst dynamics based on current prey count, starvation depends on
current population size for resource demand. `MORTALITY_ORDER` is configurable precisely
because ordering matters biologically. Batching kills would change simulation semantics.

### Next-level optimizations (large undertakings)

- Packed bit arrays (8 bools per byte) to reduce genome memory 8x
- C/Cython extensions for the genome array operations
- GPU acceleration for the phenotype computation pipeline

## Profiling Scripts

- `profile_sim.py` — cProfile of the full simulation loop
- `profile_breakdown.py` — Wall-clock timing of each phase within `run_step()`

Usage:
```
python profiler/profile_sim.py
python profiler/profile_breakdown.py
```
