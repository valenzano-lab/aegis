# Bit shuffling (breaking artificial linkage)

Under the composite genetic architecture, a genome is naturally laid out trait by trait,
and within each trait, age by age, and within each locus, bit by bit -- e.g. all of
`surv`'s loci sit next to each other, in age order, and each locus's bits sit next to each
other too.

Recombination (see `RECOMBINATION_RATE`) works by scanning the genome linearly and
deciding, bit position by bit position, whether a crossover happens between two adjacent
physical positions. This means two bits that happen to be *logically* adjacent (e.g. two
consecutive ages of `surv`, or two bits of the same locus) are, by construction, also
*physically* adjacent -- so they are more likely to be co-inherited than two arbitrary
bits, purely as an artifact of how the genome happens to be laid out. This is not
biologically meaningful linkage; it's simply an artifact of trait/age/bit ordering.

To remove this artifact, the composite architecture stores the genome in a **shuffled
physical bit order**: every individual bit (not just whole loci) is placed at a randomized
physical storage position. Recombination still scans physical position linearly, but
because the physical layout no longer correlates with the logical (trait, age,
bit-within-locus) layout, no two bits are inherently more likely to co-segregate than any
other pair, regardless of whether they belong to the same locus, the same trait, or
adjacent ages.

## How it works

- A single fixed permutation (`bit_permutation`, seeded with `seed=0`) is computed once,
  at architecture initialization, mapping every logical bit index to a physical storage
  position. It is fixed (not per-individual, not per-generation) so that all populations
  in a simulation -- including ones with different `RANDOM_SEED`s that later hybridize --
  share an identical physical genome layout.
- Genome storage, mutation, and recombination all operate in this physical (shuffled)
  order.
- Whenever a genome needs to be *interpreted* into a phenotype (i.e. whenever the
  Interpreter runs), the physical storage is first un-shuffled back into logical
  (trait x age x bit) order. This means interpretation, and therefore every phenotype
  value the simulation produces, is completely unaffected by the shuffle -- for the same
  genome content, the interpreted phenotype is identical whether or not the shuffle is
  applied. Only recombination linkage is affected.
- User-facing population-genetics output (`popgenstats`, e.g. allele frequencies,
  heterozygosity) is likewise un-shuffled back to logical order before being written, so
  what you see in the output files is indexed the way you'd expect (by trait and age), not
  by internal physical storage position.

## Parameters

There are no new parameters to set for this feature -- it is always on for the composite
genetic architecture and requires no configuration. It has no effect on the `modifying`
genetic architecture.

## What changes for you as a user

- Simulation output (phenotypes, popgen stats) is bit-for-bit identical to what you'd get
  without the shuffle, for the same genome content -- you don't need to change how you
  read or interpret output files.
- What *does* change is the linkage pattern produced by recombination over generations:
  loci are no longer artificially linked just because they're logically adjacent (e.g. two
  consecutive ages of `surv`, or two bits of one locus). If you were relying on the
  previous trait/age-ordered layout to induce or study linkage between specific loci,
  that effect is now removed.
