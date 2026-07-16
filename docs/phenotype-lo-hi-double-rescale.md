# Known issue: the lo/hi range is applied twice in the phenotype pipeline

**Status:** known, deliberately not fixed (2026-07-16). Read this before changing
`Phenotypes.clip_array_to_01`, `G_<trait>_lo` / `G_<trait>_hi`, or the default
survival/reproduction parameters.

Found while reproducing the v1 Ne / MA / AP paper under v2. Two separate problems were
compounding; one is fixed, one is documented here because the "obvious" fix is unsafe.

## The bug

`Phenotypes.clip_array_to_01` runs in the `Phenotypes` constructor and, despite its name,
*rescales* rather than clips:

```python
new_values = lo + values * (hi - lo)
```

But the architectures already own the lo/hi mapping:

- **composite** applies it itself in `compute()` (`composite/architecture.py`), and
- **modifying** doesn't use lo/hi at all — `initpheno` is the baseline in real units.

So the range gets applied a second time on top. Consequences:

- **Composite:** an all-zero genome yields surv **0.91**, not the documented floor of
  **0.70**. (`G_surv_lo`'s own docstring promises "a 50%-genome individual has surv ~0.85";
  the code actually produces ~0.955.) Every composite run is sitting on an inflated
  survival baseline.
- **Modifying:** `zeropheno` (nominally "the phenotype of an all-zero genome") is rescaled
  to `lo` (0.70) *before* `compute()` adds `initpheno` (0.95) → 1.65 → clipped to 1.0.
  With `G_surv_lo=0.7` there is **no headroom**, so the phenomap — AP or MA — can never
  move survival off 1.0. Survival appears permanently flat at 1.0.

## Why it is not fixed

Making it a true clip is correct per the documentation, but **the default parameter regime
is implicitly calibrated to the inflated survival**. With the fix, a 50%-genome individual
drops from surv ~0.955 to the documented ~0.85, and the *default* composite population
goes **extinct by ~step 30** (reproducible: `tests/functional/test_zcontainer.py` passes
31/31 on `v2`, and fails with the fix as the population dies out).

A correct fix therefore is not a one-liner. It requires:

1. changing `clip_array_to_01` to actually clip to [0, 1];
2. re-tuning the default surv / repr / reproduction parameters so populations stay viable;
3. accepting that all historical composite results shift, and saying so in the changelog.

That is a research decision with no deadline, so it has been left alone deliberately
rather than fixed by halves.

## If you hit this

Under the **modifying** architecture, lo/hi are not the range mechanism. Set them out of
the way and drive the trait from `initpheno`:

```yaml
G_surv_lo: 0.0
G_surv_hi: 1.0
G_surv_initpheno: 0.85     # survival baseline
G_repr_lo: 0.0
G_repr_hi: 1.0
G_repr_initpheno: 0.15     # reproduction baseline
G_neut_initgeno: 0.5       # start from a distribution around mid, not all-off
```

Note that `G_repr_hi` no longer acts as a hard ceiling on the reproduction rate this way;
keep the baseline low and phenomap magnitudes small to stay in range.

Also: with `G_neut_initgeno: 0` the driver loci start empty, so age-structure only emerges
over the full run — short smoke tests look flat even when everything is wired correctly.
`initgeno: 0.5` starts traits as a per-individual distribution around the midpoint, so the
signal is visible immediately and selection can push traits up *or* down.

## Related fix (shipped)

`PHENOMAP_SPECS` — the explicit `[source, site, trait, age, magnitude]` phenolist that v1
exports, used by the Ne/MA/AP paper configs — was **never read anywhere in the engine**.
Only the `PHENOMAP` dict shorthand was, which *stochastically generates* a map. Any config
carrying an explicit map therefore ran with an empty phenomap and zero pleiotropy, silently.
`ModifyingArchitecture` now consumes `PHENOMAP_SPECS` when non-empty.
