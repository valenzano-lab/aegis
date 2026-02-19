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

## Remaining Bottlenecks (hard config)

### Phenotype computation in `architect.__call__()` — 50.7% of step time

Called during hatching for all eggs. The pipeline is:

1. `ploider.diploid_to_haploid()` — logical_or + logical_xor on (N, 2, 2000, 1) bool arrays
2. `GPM.phenodiff_accelerated()` → `apply_phenolist_numba()` — loops over phenolist entries, accumulating effects per individual. Currently sequential over the phenolist dimension.
3. `Phenotypes.gaussian_smoothing()` → `gaussian_smooth_rows_with_padding_numba()` — 1D convolution per individual per trait.
4. `Phenotypes.clip_array_to_01()` — lo/hi rescaling per trait.

Potential improvements:
- `apply_phenolist_numba`: add `parallel=True` over individuals, or convert the phenolist into a sparse matrix and use `scipy.sparse` dot product.
- `gaussian_smooth_rows_with_padding_numba`: add `parallel=True` over rows (individuals are independent).
- Batch the clip operation into a single vectorized call instead of looping over trait names.

### Reproduction — 31.7% of step time

With `MUTATION_METHOD=by_index` and `BITS_PER_LOCUS=1`, mutation is already efficient.
The remaining cost is pairing (random indexing), recombination (now optimized), and
offspring genome assembly via `np.concatenate`.

### Mortalities — 10.4% of step time

`phenotypes.extract()` is called once per mortality source. It creates a fresh
`np.zeros` array and fills it via fancy indexing each time. Caching or batching
extractions for multiple mortality sources in a single pass could help.

## Profiling Scripts

- `profile_sim.py` — cProfile of the full simulation loop. Shows cumulative and self time for all functions.
- `profile_breakdown.py` — Wall-clock timing of each phase within `run_step()`. Quick way to see where time goes.

Usage:
```
python profiler/profile_sim.py
python profiler/profile_breakdown.py
```
