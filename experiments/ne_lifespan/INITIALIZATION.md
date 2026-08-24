# How the simulations start

Every result in this project descends from **one** initialisation. A single ancestral
population is created at step 0, run to equilibrium, and then branched into every treatment —
so nothing downstream can be an artefact of different starting conditions.

---

## 1. The config

```yaml
# --- population and environment -----------------------------------
INITIAL_POPULATION_SIZE: 3000
RESOURCE_MAXIMUM_AMOUNT:  3000     # carrying capacity K
RESOURCE_ADDITIVE_GROWTH: 3000     # resources regenerate to K each step
REPRODUCTION_REGULATION:  true     # population is birth-capped at K
STARVATION_PENALTY:       0.0      # no starvation mortality
AGE_LIMIT:                30       # nobody survives past age 30
MATURATION_AGE:           6        # reproduction starts at age 6
MAX_OFFSPRING_NUMBER:     3        # offspring ~ Binomial(3, repr)

# --- genome ---------------------------------------------------------
GENARCH_TYPE:      composite       # one locus per (trait, age)
BITS_PER_LOCUS:    20
PLOIDY:            2               # (engine default) diploid
REPRODUCTION_MODE: asexual
RECOMBINATION_RATE: 0              # clonal: no recombination

# --- traits ---------------------------------------------------------
G_surv_evolvable:   true           # survival evolves, age by age
G_surv_agespecific: true
G_surv_interpreter: binary
G_surv_initgeno:    0.833
G_surv_lo:          0.4523         # see §4 — effective floor is 0.70
G_surv_hi:          1.0

G_repr_evolvable:   false          # reproduction is FIXED, does not evolve
G_repr_initpheno:   0.25

G_neut_evolvable:   true           # neutral marker: no phenotypic effect
G_neut_agespecific: true
G_neut_interpreter: single_bit
G_neut_initgeno:    0.5

# --- mutation --------------------------------------------------------
G_muta_initpheno: 1.7e-4           # per site, per generation
MUTATION_RATIO:   0.1              # 1->0 favoured 10:1

# --- space ------------------------------------------------------------
LATTICE_MODE:           true       # hexagonal toroidal lattice
LATTICE_TARGET_DENSITY: 0.3        # -> 10,000 cells for K = 3,000
MIGRATION_RATE:         0.01       # move to an adjacent empty cell
MIGRATION_LONG_RATE:    0.1        # rare dispersal anywhere on the lattice

RANDOM_SEED: 1                     # seeds 1, 2, 3 were all run
```

---

## 2. What happens at step 0

`Population.initialize()` builds 3,000 individuals:

**Genomes.** Every bit is drawn **independently** as `Bernoulli(initgeno)`, on both chromatids —
`p = 0.833` for survival bits, `p = 0.5` for neutral bits. So the founding population is
**genetically variable from the outset, not clonal**. There is standing variation for selection
to act on immediately.

**Ages.** Drawn **uniformly at random over 0–29**, not a cohort of newborns. The age distribution
starts flat and relaxes to its stationary shape over the first few dozen steps.

**Phenotype.** Mean survival is **0.95 at every age** — a flat schedule, i.e. **no aging at the
start**. Aging is not built in; it has to evolve. (0.95 = 0.70 + 0.30 × 0.833, and this holds
however the bits are weighted, because the weights sum to 1.)

**Position.** Each individual gets a unique cell on the lattice; the rest stay empty at 30%
occupancy.

**Also initialised:** births = 0, body size = 0, no infection, lineage IDs issued.

---

## 3. Then: 430,000 steps of burn-in

Run under constant, permanent-water conditions with no treatment applied, until the population
has *forgotten its initial conditions*.

The stopping rule is objective rather than eyeballed. The **neutral locus** carries no
phenotypic effect, so it relaxes under mutation and drift alone toward
`p* = MUTATION_RATIO/(1 + MUTATION_RATIO) = 0.0909`, and `p*` does not depend on the mutation
rate — only the speed does. Once the neutral load sits at `p*`, the starting state has been
erased.

Measured: it started at 0.5, and reached **0.0908 against a target of 0.0909** by step 430,000
(`runs/check_equilibration.py`). Generation time works out at **15.8 steps**, so that is roughly
27,000 generations.

The ancestor that every experiment branches from therefore has:

| | |
|---|---|
| population size | 3,000 (exactly — birth-capped) |
| evolved lifespan | **20.9 steps** |
| survival at age 0 | ~0.98 |
| survival at age 28 | ~0.74 — **aging has evolved** |
| neutral locus | at equilibrium |

---

## 4. Two details that trip people up

**The survival floor is 0.70, not 0.4523.** The `[lo, hi]` rescale is applied twice in the
current code, so `G_surv_lo` is *pre-compensated*: `0.4523 → 0.4523 + 0.5477×0.4523 = 0.7000`.
The effective range is `[0.70, 1.00]`, and the data confirms it — the observed minimum across
every run is exactly 0.7000.

**Bit effects are NOT uniform.** The `binary` interpreter reads each 20-bit locus as a binary
number, so bit 0 is worth **0.150** of survival and bit 19 is worth **0.0000003** — a factor of
524,288 across the locus. (The comment in `ne_ma_ap_configs.py` claiming a uniform 0.015 per bit
is wrong; the interpreter's own docstring says "Position-dependent.") This matters when reasoning
about which mutations selection can *see*: the encoding supplies a geometric ladder of effect
sizes spanning six orders of magnitude.

---

## 5. What this means for reading the results

- **Aging is an outcome, not an assumption.** Survival starts flat at 0.95 across all ages.
- **All aging here is survival aging.** Reproduction is fixed at 0.25 and cannot evolve.
- **Only survival, mutation and the neutral marker evolve.**
- **Every treatment shares this one ancestor**, so between-arm differences are caused by the
  treatment and nothing else.
