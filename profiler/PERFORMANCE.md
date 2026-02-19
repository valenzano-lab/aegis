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

## Results by Configuration

### Default config (composite, BPL=1, pop ~550)

Per-step time: 4.93ms → 0.92ms (5.3x overall speedup)

| Phase          | Before | After  |
|----------------|--------|--------|
| Reproduction   | 80.8%  | 69.8%  |
| Mortalities    | 7.9%   | 14.8%  |
| Recording      | 8.7%   | 2.5%   |
| Hatching       | 2.1%   | 10.5%  |
| Aging          | 0.5%   | 2.3%   |

### Hard config: nemaap (modifying, sexual, 2000 loci, R=25600, pop ~6000)

Per-step time: 16.05ms → 13.74ms (1.2x speedup, ~1.3 hours saved on a 2M-step run)

| Phase          | Time % | Avg (ms) | Notes                                      |
|----------------|--------|----------|--------------------------------------------|
| Hatching       | 50.7%  | 6.82     | `architect.__call__()` phenotype computation |
| Reproduction   | 31.7%  | 4.27     | Recombination + mutation + pairing          |
| Mortalities    | 10.4%  | 1.40     | 5 sources × phenotype extraction            |
| Aging          | 6.5%   | 0.87     | Age increment + phenotype recomputation     |
| Recording      | 0.7%   | 0.10     | Buffered, negligible                        |

Extrapolated 2M-step runtime: 8.9h → 7.6h.

## Detailed Bottleneck Analysis (hard config)

cProfile self-time breakdown (100 steps, pop growing from ~1600 to ~6700):

| Rank | Function                      | Self (s) | % of 1.74s | Location                          |
|------|-------------------------------|----------|------------|-----------------------------------|
| 1    | `ploider.diploid_to_haploid`  | 0.349    | 20.0%      | `submodels/genetics/ploider.py`   |
| 2    | `gpm.phenodiff_accelerated`   | 0.337    | 19.3%      | `submodels/genetics/modifying/gpm.py` |
| 3    | `genomes.keep`                | 0.188    | 10.8%      | `dataclasses/genomes.py`          |
| 4    | `recombination_via_pairs`     | 0.167    | 9.6%       | `submodels/reproduction/recombination.py` |
| 5    | `genomes.add`                 | 0.103    | 5.9%       | `dataclasses/genomes.py`          |
| 6    | `genomes.get`                 | 0.098    | 5.6%       | `dataclasses/genomes.py`          |
| 7    | `astype` (numpy)              | 0.063    | 3.6%       | type conversions                  |
| 8    | `pairing`                     | 0.056    | 3.2%       | `submodels/reproduction/pairing.py` |

### #1: `ploider.diploid_to_haploid` — 20.0%

Called once per step during hatching via `architect.__call__()`.
Converts diploid genomes (N, 2, 2000, 1) to haploid (N, 2000, 1).

Current code:
```python
arr = np.logical_or(loci[:, 0], loci[:, 1]).astype(np.float64)
is_heterozygous = np.logical_xor(loci[:, 0], loci[:, 1])
arr[is_heterozygous] = self.DOMINANCE_FACTOR
```

The `.astype(np.float64)` creates a full copy of the array as float64 (8x the memory
of bool). Then `logical_xor` creates another full bool array. Then fancy indexing with
the heterozygous mask does scattered writes.

Potential improvements:
- Use `np.where` to avoid the intermediate bool array and scattered write:
  `arr = np.where(is_heterozygous, DOMINANCE_FACTOR, logical_or_result)`
- Use float32 instead of float64 (halves memory bandwidth)
- Fuse the logical_or and logical_xor into a single pass with numba

### #2: `gpm.phenodiff_accelerated` — 19.3%

Called once per step during hatching. The numba kernel `apply_phenolist_numba` loops
over phenolist entries (outer) and individuals (inner):

```python
for i in range(n_phenos):        # ~3000 phenolist entries for MAAP
    for j in range(n_individuals):  # ~6000 individuals
        phenodiff[j, phenotype_indices[i]] += vec_states[j, i] * magnitudes[i]
```

This is ~18M scalar operations per step. The loop order is column-major (iterating
individuals in the inner loop) which is good for cache locality on `phenodiff` rows,
but `vec_states` access pattern is strided.

Potential improvements:
- Pre-build a sparse matrix from the phenolist (once at init) and use
  `scipy.sparse.csr_matrix.dot()` — this would replace the double loop with
  optimized BLAS-backed sparse matrix multiplication
- Add `parallel=True` and `prange` over individuals
- Group phenolist entries by `phenotype_indices[i]` to reduce scattered writes

### #3: `genomes.keep` — 10.8%

Called ~256 times per step (once per `_kill` call across all mortality sources).
Each call does `self.array = self.array[individuals]` which is a fancy-indexed copy
of a (N, 2, 2000, 1) bool array.

With ~6000 individuals and 2000 loci, each genome array is ~24MB. Fancy indexing
creates a new array each time.

Potential improvements:
- Batch all mortality sources into a single kill mask, then call `keep` once
  instead of ~2-6 times per step. This is the biggest structural win here.
- Use a boolean mask instead of integer indices (avoids index array allocation)

### #4: `recombination_via_pairs` — 9.6%

Already optimized. The remaining 167ms/100 steps is mostly numpy overhead
(reshape, copy, binomial, random integers) around the numba kernel.

### #5-6: `genomes.add` + `genomes.get` — 11.5% combined

`add` does `np.concatenate` on large bool arrays during reproduction (merging
offspring into population). `get` does fancy indexing to extract parental genomes.

These are fundamentally memory-bandwidth-bound operations on large arrays.

Potential improvements:
- Pre-allocate a genome buffer with max capacity and use views instead of
  concatenation. This avoids repeated allocation/copy cycles.

### #7: `astype` — 3.6%

Scattered across the codebase. The `Genomes.__init__` always calls
`.astype(np.bool_)` even when the input is already bool. Adding a dtype check
(`if array.dtype != np.bool_: array = array.astype(np.bool_)`) would skip
unnecessary copies.

### #8: `pairing` — 3.2%

Sexual pairing logic with random gamete selection. Uses fancy indexing on genome
arrays. Hard to optimize further without restructuring the reproduction pipeline.

### Summary: where to focus next

The top 3 targets for further optimization are:

1. **Batch mortality kills** — call `_kill` once per step instead of per-source.
   Would reduce `genomes.keep` calls from ~256 to ~1. Estimated saving: ~8-10%.

2. **Sparse matrix for phenomap** — replace `apply_phenolist_numba` with a
   precomputed sparse matrix multiply. Estimated saving: ~10-15%.

3. **Optimize diploid_to_haploid** — fuse operations, use float32, avoid
   intermediate arrays. Estimated saving: ~5-10%.

## Profiling Scripts

- `profile_sim.py` — cProfile of the full simulation loop. Shows cumulative and self time for all functions.
- `profile_breakdown.py` — Wall-clock timing of each phase within `run_step()`. Quick way to see where time goes.

Usage:
```
python profiler/profile_sim.py
python profiler/profile_breakdown.py
```
