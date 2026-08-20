# selectable_via_surv

`selectable_via_surv` is a genetic trait -- with its own genome loci, heritable, mutable,
and recombinable just like `surv` or `repr` -- whose phenotype is used to compute a
selection-derived correction that gets added to the `surv` phenotype. It lets you impose
an arbitrary selection pressure (directional or stabilizing) on an evolving trait, mediated
through survival.

It is a first-class trait: it has its own phenotype, computed the same way as any other
trait (loci -> interpreter -> `[lo, hi]` scaling -> optional smoothing), and can be
inspected on its own (e.g. via `phenotypes.extract("selectable_via_surv", ages)`) in
addition to its effect on `surv`.

It is disabled by default (`G_selectable_via_surv_evolvable: false`), so existing configs
and simulations are unaffected unless you explicitly opt in.

## How it works

1. `selectable_via_surv`'s phenotype value is computed exactly like any other trait's:
   from its own genome loci, through its configured interpreter, scaled to
   `[G_selectable_via_surv_lo, G_selectable_via_surv_hi]`.
2. That value is passed through one of two selection functions
   (`G_selectable_via_surv_mode`) to produce a correction ("delta").
3. The delta is added to the `surv` phenotype.
4. The resulting `surv` phenotype is hard-clipped to `[0, 1]` (not to `G_surv_lo`/`G_surv_hi`
   -- an absolute `[0, 1]` cap, since `surv` is used directly as a survival probability).

### Directional selection (`G_selectable_via_surv_mode = "directional_selection"`)

A step function: individuals at or above a cutoff get a constant benefit; individuals
below the cutoff get a constant (typically negative) penalty.

```
delta = directional_sel_constant_benefit   if trait_value >= directional_sel_cutoff_value
delta = directional_sel_constant_penalty   otherwise
```

Note: the comparison is `>=`, so a trait value exactly equal to the cutoff counts as a
benefit, not a penalty.

### Stabilizing selection (`G_selectable_via_surv_mode = "stabilizing_selection"`)

A Gaussian centered on a target value, valid within a configurable number of standard
deviations; beyond that range, a constant penalty applies instead of the (fast-decaying)
Gaussian tail.

```
z = (trait_value - stabilizing_sel_mean) / stabilizing_sel_sd

delta = stabilizing_sel_max_benefit * exp(-0.5 * z**2)   if |z| <= stabilizing_sel_const_penalty_beyond_sd
delta = stabilizing_sel_const_penalty                     otherwise
```

The Gaussian peaks exactly at `stabilizing_sel_mean`, where `delta == stabilizing_sel_max_benefit`,
and decays with distance from the mean. `stabilizing_sel_const_penalty_beyond_sd` sets how
many standard deviations out the Gaussian is used before switching to the flat penalty
(e.g. `2` means the Gaussian applies within 2 standard deviations of the mean, and the
constant penalty applies beyond that).

## Age-specificity

`G_selectable_via_surv_agespecific` controls whether the trait (and its correction) varies
by age, same as any other trait:

- `false` (default): a single trait value per individual. Its delta is broadcast equally to
  every age of `surv`.
- `true`: one trait value per age (matching `AGE_LIMIT`). Its delta is applied to the
  matching age of `surv` -- this requires `selectable_via_surv`'s length to exactly match
  `surv`'s length (i.e. `G_surv_agespecific` must also be `true`, or both must resolve to
  the same number of loci). A mismatched, non-broadcastable configuration raises a
  `ValueError` at simulation startup.

## Parameters

### Genome encoding (same fields every trait has)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `G_selectable_via_surv_evolvable` | bool | `false` | Turns the trait on. When `false`, it has no genome loci and no effect on `surv` at all. |
| `G_selectable_via_surv_agespecific` | bool | `false` | `true` = one locus (and one correction) per age; `false` = a single value broadcast to all ages. |
| `G_selectable_via_surv_interpreter` | str | `"uniform"` | Any standard interpreter (`uniform`, `binary`, `linear`, `custom_weighted`, ...). See [custom-weighted-interpreter.md](custom-weighted-interpreter.md) for `custom_weighted`. |
| `G_selectable_via_surv_initgeno` | float | `0.5` | Initial genotype frequency (probability a genome bit starts as 1). |
| `G_selectable_via_surv_custom_weights` | list | `None` | Only used when `G_selectable_via_surv_interpreter` is `"custom_weighted"`. |
| `G_selectable_via_surv_lo` | float | `0` | Minimum value of the trait's own phenotype. |
| `G_selectable_via_surv_hi` | float | `1` | Maximum value of the trait's own phenotype. |

### Selection function

| Parameter | Type | Default | Description |
|---|---|---|---|
| `G_selectable_via_surv_mode` | str | `"directional_selection"` | `"directional_selection"` or `"stabilizing_selection"`. |
| `G_selectable_via_surv_directional_sel_cutoff_value` | float | `0.5` | Directional mode: trait value at or above which the benefit applies. |
| `G_selectable_via_surv_directional_sel_constant_benefit` | float | `0` | Directional mode: constant added to `surv` when trait value >= cutoff. |
| `G_selectable_via_surv_directional_sel_constant_penalty` | float | `0` | Directional mode: constant added to `surv` when trait value < cutoff. Typically negative. |
| `G_selectable_via_surv_stabilizing_sel_mean` | float | `0.5` | Stabilizing mode: mean (peak) of the Gaussian. |
| `G_selectable_via_surv_stabilizing_sel_sd` | float | `0.1` | Stabilizing mode: standard deviation of the Gaussian. Must be positive. |
| `G_selectable_via_surv_stabilizing_sel_max_benefit` | float | `0` | Stabilizing mode: value added to `surv` at the mean (the Gaussian's peak height). |
| `G_selectable_via_surv_stabilizing_sel_const_penalty` | float | `0` | Stabilizing mode: constant added to `surv` beyond the standard-deviation threshold. Typically negative. |
| `G_selectable_via_surv_stabilizing_sel_const_penalty_beyond_sd` | float | `2` | Stabilizing mode: number of standard deviations from the mean within which the Gaussian applies. Must be `>= 0`. |

## Examples

### Directional selection

Individuals with a high `selectable_via_surv` value get a small survival boost; low
values get a penalty:

```yaml
G_selectable_via_surv_evolvable: true
G_selectable_via_surv_agespecific: false
G_selectable_via_surv_interpreter: uniform
G_selectable_via_surv_mode: directional_selection
G_selectable_via_surv_directional_sel_cutoff_value: 0.5
G_selectable_via_surv_directional_sel_constant_benefit: 0.02
G_selectable_via_surv_directional_sel_constant_penalty: -0.02
```

### Stabilizing selection

Individuals near a target value of `0.5` get a survival boost that fades with distance;
individuals more than 2 standard deviations away get a flat penalty:

```yaml
G_selectable_via_surv_evolvable: true
G_selectable_via_surv_agespecific: true
G_selectable_via_surv_interpreter: uniform
G_selectable_via_surv_mode: stabilizing_selection
G_selectable_via_surv_stabilizing_sel_mean: 0.5
G_selectable_via_surv_stabilizing_sel_sd: 0.15
G_selectable_via_surv_stabilizing_sel_max_benefit: 0.02
G_selectable_via_surv_stabilizing_sel_const_penalty: -0.05
G_selectable_via_surv_stabilizing_sel_const_penalty_beyond_sd: 2
```

## Notes

- `surv` is always hard-clipped to `[0, 1]` after the correction is applied, regardless of
  `G_surv_lo`/`G_surv_hi`.
- This feature is only meaningful under the composite genetic architecture (like
  `custom_weighted`, see [custom-weighted-interpreter.md](custom-weighted-interpreter.md)).
  Under the `modifying` architecture, `selectable_via_surv` has no genome loci, so it
  behaves as an inert constant (`G_selectable_via_surv_initpheno`, default `0.0`) and
  contributes no correction to `surv`.
