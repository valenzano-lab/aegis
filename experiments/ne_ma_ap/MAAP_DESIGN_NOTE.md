# MAAP resubmission: design note

For the meeting between Dario, Martin and Ruchitha on the MAAP resubmission. The aim is to
settle the design so the runs can start straight afterwards.

## Goal

Replicate Martin's results (Bagic & Valenzano, bioRxiv 2022) in the current AEGIS, with one
change in principle: **both arms start from a pre-evolved population at equilibrium**, and
**MA accumulates deleterious variants only**, as it did in the paper.

Traits are staged. Martin's work covered survival; Ruchitha takes reproduction.

1. Survival evolves, reproduction fixed (replicates the paper).
2. Reproduction evolves, survival fixed.
3. Survival and reproduction co-evolve.

## Proposed design

**Phase 1: burn-in.** Start from initialization. Bits mutate in both directions (0→1 and
1→0) in the proportion set by `MUTATION_RATIO`. A few burn-in scenarios, like the ones we use
for the AP model. Stop when the neutral locus reaches mutation–drift equilibrium, as in
Bradshaw et al. (bioRxiv). In the seed-1 runs that took about 400,000–500,000 steps at every
Ne tested.

**Phase 2: split the equilibrated checkpoint into two arms.** Both arms resume the *same*
checkpoint, so they differ only in the regime, not in their history.

| | MA arm | AP arm |
|---|---|---|
| Pleiotropy | none: each bit string affects only itself | pleiotropic drivers as in our AP model |
| Mutation direction | 1→0 only | both directions, per `MUTATION_RATIO` |
| Beneficial variants | impossible | possible, early benefit with a late cost |

## What the engine already supports

- **1→0-only mutation:** `MUTATION_RATIO: 0` gives exactly this (0→1 rate = 0, 1→0 rate = 1).
  No engine change needed. But the ratio is global, so the neutral locus in the MA arm will
  also decay one way, and it stops being usable as a neutral clock after the split. The AP
  arm keeps it.
- **Resuming a checkpoint under a different regime:** `aegis sim -r --override KEY=VALUE`
  exists, with tests, but only on topic branches (`feat-resume-param-override`,
  `exp-ne-lifespan`). **It is not in `v2` yet.** Merge it before the sweep.
- **Equilibrium check:** `runs/check_equilibration.py` tests a finished run offline. The
  burn-in needs either a fixed length set from pilot runs, or a stop condition inside the
  engine.

## What still needs work

- **Changing the pleiotropy map at the split.** `--override` parses scalar values only, so it
  cannot remove or replace `PHENOMAP_SPECS` (a list). I also still need to check whether the
  phenomap is rebuilt from the parameters on resume or restored from the checkpoint. This
  only matters if the burn-in includes pleiotropy (question 1 below).
- **Reproduction arms.** Reproduction is currently fixed (`G_repr_evolvable: False`). Until
  the lo/hi double-rescale is fixed in the engine, making it evolvable needs the compensated
  range (`G_repr_lo: 0, G_repr_hi: 0.7071` for an effective [0, 0.5]).

## Questions to decide at the meeting

1. **Burn-in scenarios.** Are the "scenarios typical of the AP model" different pleiotropy
   settings, different mutation ratios, or both?
   - Burn-in *with* pleiotropy: the MA arm must drop it at the split, so phenotypes jump at
     once; or keep the existing drivers frozen.
   - Burn-in *without* pleiotropy: the split is clean, but every AP effect arises after it.
2. **AP driver direction.** Martin's AP built in the Williams trade-off (early benefit, late
   cost). Our current AP assigns half the drivers each direction, which *tests* the trade-off
   instead. Do we copy Martin's setup, keep the test, or run both?
3. **The replication spec.** Which of Martin's figures, which Ne values, run lengths and
   number of seeds? That list is the sweep.
4. **Reproduction baseline.** What fixed reproduction value do the survival-only runs use,
   and what fixed survival do the reproduction-only runs use?
5. **Compute.** Ne=30,000 took 64 hours for one run and added little over Ne=3,000 in
   selection efficiency. Do we drop it, or keep it for fewer seeds?

## Future direction (not for MAAP): an evolvable pleiotropy map

In MAAP the connection map is fixed by us. The long-term version lets it evolve: each bit
string may affect only itself (no pleiotropy) or also other survival, reproduction or neutral
bit strings, and which ones it affects is itself heritable and mutable.

**Why it's worth doing.** It follows the same principle as the current model, where a
mutation's fitness effect comes from where it lands and not from a distribution we choose.
How much pleiotropy exists becomes a result instead of an assumption. It also asks a sharp
question about aging: selection should favour modifiers that uncouple an early benefit from
its late cost, but selection weakens with age, so the pressure to remove late costs is also
weak. The prediction is that early benefits get tuned while late costs persist, so
antagonistic pleiotropy is what selection *can't* clean up rather than something we build in.

**Link to Ne.** Changing a connection is a second-order effect, so selection on it is weak,
and small populations should be unable to reshape their connections while large ones can
(Lynch's drift-barrier argument applied to pleiotropy).

**Biology to anchor it:**
- Gene regulatory networks are sparse, and the number of genes each gene affects is highly
  uneven (Barabási & Oltvai 2004).
- Most genes affect few traits (Wagner et al. 2008; Wang, Liao & Zhang 2010).
- Pleiotropy varies genetically: some loci change how strongly two traits are coupled
  (Pavlicev et al. 2008).
- Modularity can evolve from a cost of connections (Clune et al. 2013) or from changing
  environments (Kashtan & Alon 2005).

**Design risks:**
1. Gaining and losing connections needs its own mutation bias, and results may depend on it
   strongly. It has to be varied, not fixed.
2. A full map over all traits and ages is tens of thousands of possible connections per
   individual. A workable option: each locus has a few fixed slots, each holding a mutable
   target and weight, all encoded in bits.
3. Go in steps: evolvable weights on fixed sparse wiring first, then connections that can be
   gained or lost.
4. Connections with no effect will drift, so they need a neutral control, as the neutral
   locus is for the genome.
5. MA and AP stop being two arms and become ends of a continuum. That's the more interesting
   result, but it belongs in a separate paper, with MAAP's fixed-map runs as the reference.
