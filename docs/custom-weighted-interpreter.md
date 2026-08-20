# Custom-weighted interpreter

`custom_weighted` is an interpreter type (composite genetic architecture only) that lets
you assign an arbitrary, user-defined weight to each bit of a locus, instead of using one
of the built-in interpreters (`binary`, `linear`, `uniform`, ...). This is useful when you
want individual bits to have a specific, unequal effect size on the trait -- e.g. a "major
effect" bit alongside several "minor effect" bits.

## How it works

For a locus with `BITS_PER_LOCUS` bits, you supply a list of `BITS_PER_LOCUS` non-negative
weights. The weights are normalized (divided by their sum) and dot-multiplied with the
locus's bits (1 = present, 0 = absent):

```
trait_value = sum(bit_i * weight_i) / sum(weights)
```

For example, with weights `[10, 3, 1, 1]` and bits `[True, False, True, False]`:

```
trait_value = (1*10 + 0*3 + 1*1 + 0*1) / (10+3+1+1) = 11/15 ≈ 0.733
```

This raw value in `[0, 1]` is then mapped to `[G_<trait>_lo, G_<trait>_hi]` like any other
interpreter's output.

## Parameters

Set per trait (`<trait>` is one of `surv`, `repr`, `muta`, `neut`, `grow`,
`selectable_via_surv`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `G_<trait>_interpreter` | str | trait-specific | Set to `"custom_weighted"` to activate this interpreter for the trait. |
| `G_<trait>_custom_weights` | list of numbers | `None` | Bit weights, one per bit in the locus. Only used when `G_<trait>_interpreter` is `"custom_weighted"`. |

## Validation

When `G_<trait>_interpreter` is `"custom_weighted"`:

- `G_<trait>_custom_weights` must be a list.
- It must contain exactly `BITS_PER_LOCUS` numbers.
- All weights must be non-negative numbers.
- The weights must sum to a positive total.

Violating any of these raises a `ValueError` at simulation startup.

## Example

```yaml
BITS_PER_LOCUS: 4
G_surv_interpreter: custom_weighted
G_surv_custom_weights: [10, 3, 1, 1]
```

This makes the first bit of the `surv` locus a "major effect" bit (weight 10) and the
remaining three bits "minor effect" bits (weight 1-3 each).

## Interaction with bit shuffling

`custom_weighted` is bit-position-sensitive: weight `i` always applies to logical bit `i`
of the locus (the order you'd naturally think of the bits in), regardless of how the
genome is physically laid out in storage. See [bit-shuffling.md](bit-shuffling.md) -- the
physical bit shuffle only affects storage/recombination linkage, never interpretation, so
this holds true independent of that feature.
