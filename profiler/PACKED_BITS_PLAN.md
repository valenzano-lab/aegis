# Packed Bit Genomes — Implementation Plan

## Goal

Replace the current `np.bool_` genome storage (1 byte per bit) with `np.uint8`
packed storage (8 bits per byte). This reduces genome memory 8x, making every
array operation (copy, index, concatenate) 8x cheaper and enabling population
sizes of 50,000–100,000 individuals on a single machine.

## Current State

- Genome array: `np.bool_` with shape `(n_individuals, ploidy, n_loci, bits_per_locus)`
- At 6,000 individuals, 2 ploidy, 2,000 loci, 1 BPL: **24 MB**
- At 60,000 individuals: **240 MB** — every `keep`/`add`/`get` copies most of this

## Target State

- Genome array: `np.uint8` with shape `(n_individuals, ploidy, n_packed_bytes)`
- Where `n_packed_bytes = ceil(n_loci * bits_per_locus / 8)`
- At 6,000 individuals, 2 ploidy, 2,000 loci, 1 BPL: **3 MB** (250 packed bytes)
- At 60,000 individuals: **30 MB**

---

## Risks and Mitigations

### R1: Padding bits corrupt mutation/recombination
**Risk**: If total bits (n_loci × bits_per_locus) isn't divisible by 8, `np.packbits`
pads with zeros. Mutations targeting padding bits silently have no effect, changing
evolutionary dynamics.
**Mitigation**: Enforce at parameter validation time that `n_loci * bits_per_locus`
must be divisible by 8. Reject configs that violate this. This is a mild constraint —
all realistic configs already satisfy it. A future enhancement could auto-pad the
genome size to the next multiple of 8 by adding inert loci, but this is not needed
for the initial implementation.

### R2: Bit ordering mismatch
**Risk**: `np.packbits` uses big-endian bit order (MSB first). If unpack, mutation
index calculation, or recombination masking uses a different convention, bits get
silently scrambled — locus 0's bit ends up in locus 7's position. The simulation
runs but the biology is wrong.
**Mitigation**: Use `np.packbits(array, axis=-1, bitorder='big')` and
`np.unpackbits(array, axis=-1, bitorder='big')` everywhere — always explicit, never
rely on defaults. Unit tests that pack a known pattern (e.g. `[1,0,0,0,0,0,0,0]`)
and verify the packed byte is `0b10000000 = 128`. Property test: `unpack(pack(x)) == x`
for random arrays.

### R3: Recombination crossover precision
**Risk**: If the packed recombination rounds crossover points to byte boundaries,
crossover resolution drops from 1 bit to 8 bits. For BPL=1, this means crossovers
only happen every 8 loci instead of every locus — fundamentally different biology.
**Mitigation**: The recombination kernel must handle sub-byte crossovers with bitwise
masking at the boundary byte. Full-byte segments swap directly (fast), only the
boundary byte needs bit-level work. Test with crossover at every possible bit position
within a byte (0-7) and verify against the unpacked implementation.

### R4: Simulation results change (RNG stream shift)
**Risk**: If the packed implementation changes the number or order of random number
draws (e.g., mutation generates indices differently), the random stream diverges.
Same seed produces different results. Not a correctness bug but breaks reproducibility
with pre-existing published results.
**Mitigation**: The mutation and recombination code must generate the same random
numbers in the same order as the current implementation. The packed version should
unpack → generate randoms → apply to packed data, preserving the RNG sequence.
Side-by-side validation (same seed, compare every step) catches any divergence.

### R5: Checkpoint compatibility
**Risk**: Existing checkpoints store `Population` objects via pickle with bool genome
arrays. Packed genomes would break loading old checkpoints.
**Mitigation**: `Genomes.__init__` already handles both bool and uint8 input (packs
bool on construction). `Checkpoint.load` will transparently handle old bool-based
checkpoints by letting `Genomes.__init__` pack them. New checkpoints will be smaller.
Document that old checkpoints are forward-compatible.

### R6: Recording/output format changes
**Risk**: `featherrecorder`, `intervalrecorder`, and `popgenstats` expect bool arrays.
**Mitigation**: `flatten()` and `get_array()` return unpacked bool arrays — same as
today. The recording interface doesn't change. `popgenstats` receives unpacked arrays
via `genomes.get_array()` or `genomes.unpack()`.

---

## Pre-evaluation: Expected Performance

Based on current profiling (hard config, 6000 individuals, 2000 loci):

| Operation | Current (bool, 24MB) | Packed (uint8, 3MB) | Expected speedup |
|-----------|---------------------|---------------------|------------------|
| `genomes.keep` (per call) | 0.73ms | ~0.09ms | ~8x |
| `genomes.add` (per call) | 1.03ms | ~0.13ms | ~8x |
| `genomes.get` (per call) | 0.33ms | ~0.04ms | ~8x |
| `genomes.get_array` (copy) | 0.24ms | ~0.03ms | ~8x |
| Unpack for phenotype pipeline | 0ms (already bool) | ~0.3ms | new cost |
| **Net per-step estimate** | ~7.3ms | ~5-6ms | ~1.2-1.5x |

At 60,000 individuals (10x current), the savings scale linearly with array size
while the unpack cost stays proportional to population size. The crossover point
where packed bits become slower than bool is well below 100 individuals — not a
realistic scenario.

## Post-evaluation Plan

After implementation, run these benchmarks:

1. **Microbenchmark**: time `keep`, `add`, `get`, `flatten`, `pack`, `unpack`
   at 6,000 and 60,000 individuals. Compare to current bool implementation.

2. **End-to-end hard config**: 200 steps of the nemaap MAAP config (R=25600).
   Compare per-step time and phase breakdown to the 7.3ms/step baseline.

3. **Scaling test**: run 50 steps at population sizes 1K, 5K, 10K, 25K, 50K, 100K.
   Plot per-step time vs population size for both implementations. Verify that
   packed bits scales linearly while bool becomes superlinear (due to cache pressure).

4. **Correctness validation**: run the same config with both implementations for
   1000 steps with the same seed. Compare population statistics (age distribution,
   allele frequencies, population size trajectory) — they must be identical.

---

## Modules Requiring Changes

### 1. `src/aegis_sim/dataclasses/genomes.py` — Core change

Create `PackedGenomes` class (or modify `Genomes` in-place):

```
class Genomes:
    def __init__(self, array):
        if array.dtype == np.bool_:
            # Pack on construction
            self._packed = np.packbits(array, axis=-1)  # pack along last axis
            self._unpacked_shape = array.shape
        elif array.dtype == np.uint8:
            self._packed = array
            self._unpacked_shape = None  # must be set externally or inferred

    def get(self, individuals):        # index into packed array
    def keep(self, individuals):       # index into packed array
    def add(self, genomes):            # concatenate packed arrays
    def flatten(self):                 # unpack then flatten (for recording)
    def get_array(self):               # unpack and return bool copy
    def shape(self):                   # return unpacked shape
    def unpack(self):                  # return bool view for computation
    def __len__(self):                 # first dim of packed array
    def __getitem__(self, key):        # index and return new Genomes
```

Key design decision: `get`, `keep`, `add`, `__getitem__` operate on packed data
(fast). `flatten`, `get_array`, `unpack` return bool arrays (for computation and
recording). The packed array is the source of truth.

### 2. `src/aegis_sim/submodels/reproduction/mutation.py`

**`_mutate_by_index`**: Currently generates random (individual, chromatid, locus, bit)
indices and flips bits. With packed storage, need to convert (locus, bit) to
(byte_index, bit_position) and use XOR masks.

```python
byte_idx = (locus * bpl + bit) // 8
bit_pos = (locus * bpl + bit) % 8
packed[individual, chromatid, byte_idx] ^= (1 << (7 - bit_pos))
```

**`_mutate_by_bit`**: Generates random probabilities for every bit. Must unpack,
mutate, repack. This method is already slower than `by_index` — acceptable as
fallback. The hard config uses `by_index` so this is not on the critical path.

### 3. `src/aegis_sim/submodels/reproduction/recombination.py`

**`recombination_via_pairs_numba`**: Currently operates on flat bool arrays with
shape `(n_individuals, 2, n_sites)`. With packed bits, `n_sites` becomes
`n_packed_bytes`. The difference-array approach works on byte granularity for
full-byte swaps. Crossover points within a byte need bitwise masking:

```python
# For crossover at bit position p within byte b:
mask = 0xFF << (8 - p)  # bits before crossover
byte_b_c0 = (c0[b] & mask) | (c1[b] & ~mask)
byte_b_c1 = (c1[b] & mask) | (c0[b] & ~mask)
```

The numba kernel already works element-by-element, so adapting to byte-level
with bit masking at boundaries is straightforward.

### 4. `src/aegis_sim/submodels/genetics/architect.py`

**`__call__`**: Calls `genomes.get_array()` then `envdrift.call()` then
`architecture.compute()`. Change to `genomes.unpack()` — returns bool array
for the phenotype pipeline. No other changes needed.

### 5. `src/aegis_sim/submodels/genetics/ploider.py`

**`diploid_to_haploid`**: Currently takes bool array `(N, 2, loci, bpl)`.
Will receive unpacked bool array from architect — no change needed.

### 6. `src/aegis_sim/submodels/genetics/composite/architecture.py`

**`compute`**: Receives unpacked genomes from architect. No change needed.

**`init_genome_array`**: Currently returns `(N, ploidy, n_loci, bpl)` float array
that gets cast to bool by `Genomes.__init__`. No change needed — `Genomes.__init__`
will pack it.

### 7. `src/aegis_sim/submodels/genetics/modifying/architecture.py`

**`compute`**: Same as composite — receives unpacked genomes. No change needed.

**`init_genome_array`**: Same as composite. No change needed.

### 8. `src/aegis_sim/submodels/genetics/envdrift.py`

**`call`**: Does `np.logical_xor(self.map, array)` on unpacked bool arrays.
No change needed — receives unpacked array from architect.

**`evolve`**: Flips a random bit in `self.map` (bool array). No change needed —
the map is small (one genome shape, not per-individual).

### 9. `src/aegis_sim/submodels/reproduction/pairing.py`

**`_assemble_children`**: Numba kernel reads `genome_array[parent, chromatid, :, :]`
and writes to children. With packed storage, this copies packed bytes directly —
actually simpler and faster. Change `genome_array` parameter to packed uint8 array.

### 10. `src/aegis_sim/submodels/reproduction/reproduction.py`

**`generate_offspring_genomes`**: Calls recombination, pairing, mutation in sequence.
Recombination and mutation operate on raw arrays (packed). Pairing returns packed
children. `Genomes(result)` at the end packs if needed. Minimal wiring changes.

### 11. `src/aegis_sim/dataclasses/population.py`

**`__imul__`**: Calls `genomes.keep()` — no change (operates on packed data).
**`__iadd__`**: Calls `genomes.add()` — no change (concatenates packed data).
**`initialize`**: Calls `Genomes(init_genome_array(n))` — packing happens in constructor.
**`make_eggs`**: Same pattern.

### 12. `src/aegis_sim/recording/featherrecorder.py`

**`write_genotypes`**: Calls `population.genomes.flatten()` — returns unpacked bool.
No change needed.

### 13. `src/aegis_sim/recording/intervalrecorder.py`

**`write`**: Calls `population.genomes.flatten().mean(0)` — returns unpacked bool.
No change needed.

### 14. `src/aegis_sim/recording/popgenstatsrecorder.py`

**`write`**: Passes `genomes.array` to `popgenstats.calc()`. Must change to
`genomes.unpack()` or `genomes.get_array()`. The popgenstats module does extensive
reshape/mean operations that assume bool arrays.

### 15. `src/aegis_sim/utilities/popgenstats.py`

Receives genome arrays and does `.reshape()`, `.mean()`, `.sum()` operations.
Must receive unpacked bool arrays. No internal changes needed if caller unpacks.

### 16. `src/aegis_sim/checkpoint.py`

**`capture`**: Pickles the Population object (which contains Genomes). Packed
genomes will be pickled as uint8 arrays — smaller checkpoint files (8x for genome
portion).

**`load`**: Must handle both old (bool) and new (uint8) checkpoint formats.
Add a version check or detect dtype on load.

---

## Test Plan

### Layer 1: Unit tests (run on every `pytest` invocation — fast, milliseconds)

1. **Pack/unpack roundtrip**: `unpack(pack(x)) == x` for random bool arrays at
   various shapes. Test with total bits exactly divisible by 8.

2. **Known bit patterns**: pack `[1,0,0,0,0,0,0,0]` → verify byte is 128.
   Pack `[1,1,1,1,1,1,1,1]` → verify byte is 255. Ensures bit ordering is correct.

3. **Bit-level mutation**: flip specific bits in packed array via `by_index`,
   verify only target bit changed. Test at positions 0, 7, and cross-byte boundaries.

4. **Packed recombination**: crossover at byte-aligned and non-aligned positions
   (bit 0, 3, 7 within a byte). Verify result matches unpacked recombination.

5. **Packed pairing**: verify children genomes match parent chromatids when
   operating on packed data.

6. **Packed keep/add/get**: verify these produce same results as unpacking,
   operating on bool, and repacking.

7. **Divisibility-by-8 validation**: verify that configs with non-divisible genome
   sizes are rejected at parameter validation time.

### Layer 2: Legacy reference implementation (kept permanently in codebase)

Keep the current `Genomes` class as `LegacyGenomes` in a separate file
(e.g. `src/aegis_sim/dataclasses/legacy_genomes.py`). This is not used in
production — it exists solely as a reference for testing.

### Layer 3: Side-by-side validation (run in CI via `pytest -m validation` — slow, ~30-60s)

A test marked `@pytest.mark.validation` that:

1. Initializes two simulations with the same seed — one using `PackedGenomes`,
   one using `LegacyGenomes`.
2. Runs both for 100 steps.
3. After each step, unpacks the packed genomes and compares to the legacy genomes
   array. They must be identical.
4. At the end, compares population size, age distribution, and allele frequencies.
5. Runs for both composite (BPL=8) and modifying (BPL=1) architectures.
6. Runs for both sexual and asexual reproduction modes.

This test is the ultimate safety net. It catches any interaction bug between
mutation, recombination, pairing, phenotype computation, and mortality that
unit tests might miss.

### Existing tests (must pass unchanged)

All 381+ existing tests must pass. The `Genomes` public interface is preserved,
so tests using `Genomes` through its API work without modification.

---

## Implementation Order

### Phase 1: Core data structure (lowest risk, testable in isolation)
1. Implement `PackedGenomes` with pack/unpack/get/keep/add/flatten/shape
2. Write unit tests for pack/unpack roundtrip and all operations
3. Verify existing `Genomes` tests pass with `PackedGenomes`

### Phase 2: Mutation (moderate risk, self-contained)
4. Adapt `_mutate_by_index` to work on packed arrays
5. Write bit-level mutation tests
6. Verify mutation tests pass

### Phase 3: Recombination (moderate risk, self-contained)
7. Adapt `recombination_via_pairs_numba` for packed byte arrays
8. Write packed recombination tests including byte-boundary crossovers
9. Verify recombination tests pass

### Phase 4: Pairing (low risk)
10. Adapt `_assemble_children` to copy packed bytes
11. Verify pairing tests pass

### Phase 5: Integration (wiring)
12. Update `architect.__call__` to use `unpack()`
13. Update `popgenstatsrecorder` to pass unpacked arrays
14. Update `checkpoint.py` for format detection
15. Run full test suite
16. Run side-by-side simulation validation

### Phase 6: Benchmarking
17. Run microbenchmarks (keep/add/get at 6K and 60K individuals)
18. Run end-to-end hard config benchmark
19. Run scaling test (1K to 100K individuals)
20. Update PERFORMANCE.md with results
