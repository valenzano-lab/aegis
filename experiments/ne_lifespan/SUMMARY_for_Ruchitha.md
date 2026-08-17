# Is it carrying capacity or Ne? — what we ran, and what came out

**Figures:** `runs/routes_decomposition.png` · `runs/killifish_windows.png`
**Code and full record:** `experiments/ne_lifespan/` (branch `exp-ne-lifespan`), details in `HANDOFF.md`

---

## 1. Your question was right, and it was a real problem

You pointed out that in the existing sweep, "Ne" is set by `RESOURCE_MAXIMUM_AMOUNT` — the
carrying capacity. That is literally true: the config file says `# Ne is set by the resource
limit`, and `build()` sets `INITIAL_POPULATION_SIZE = RESOURCE_MAXIMUM_AMOUNT =
RESOURCE_ADDITIVE_GROWTH = ne`. So the sweep never varied Ne independently of K, and there was
no measurement of Ne to appeal to.

Your proposed fix — use N·u to decouple things — needed one correction, which Dario spotted
straight away: **N·u and Ne are different quantities.** N·u is the *supply* of new variants;
Ne is *selectability*, whether selection can resolve a variant at all (|s| vs 1/2Ne). Changing
the mutation rate µ does not change Ne. So "vary µ to vary Ne" varies supply, not drift.

But the instinct was correct, and it turns out K reaches life history by **three** distinct
paths, not two:

| | route | what it changes |
|---|---|---|
| **1** | K → N → **Ne** | the drift barrier: can selection *see* a late-acting variant? |
| **2** | K → **N·u** | mutational supply: how much raw material arrives? |
| **3** | K → resource-limited **mortality** | Williams/Medawar: how steep is the selection gradient? |

The original design moved all three at once. The measured effect was their sum.

---

## 2. What we built

The hard part is varying Ne while holding K fixed. In a well-mixed AEGIS population that is
impossible: offspring number is drawn from a binomial, which cannot be overdispersed, so Ne is
pinned within about 2× of census N.

**Spatial structure breaks that.** With `LATTICE_MODE`, mating and offspring placement are
local, so drift is governed by neighbourhood size rather than global census — which can sit far
below N. That is population fragmentation, and it is also biologically the right model for
killifish pools.

We calibrated it first rather than assuming (12 short runs): `MIGRATION_LONG_RATE` turned out to
be the strong knob, mapping onto spatial structure F_ST over a **15× range** while census N held
at exactly 3000 in every arm.

Then: **one** pre-evolved ancestor (430,000 steps, equilibrated on the neutral locus), branched
into every treatment. Because all arms share that ancestor, nothing downstream can be burn-in
history. Three seeds throughout.

---

## 3. Result — all three routes are real

![routes](../../runs/routes_decomposition.png)

Against the control (evolved lifespan **20.78 ± 0.16**):

| route | manipulation | Δ lifespan | Δ early survival | Δ late survival | late/early |
|---|---|--:|--:|--:|--:|
| 1 drift barrier | F_ST 0.09 → 0.68 | −2.28 | −0.0055 | −0.0261 | 4.7× |
| 2 mutational supply | µ × 4 | −6.05 | −0.0262 | −0.1003 | 3.8× |
| 3 extrinsic mortality | starvation deaths | −4.38 | −0.0118 | −0.1236 | 10.5× |

**The answer to your question is not "Ne" and not "K" — it is that K acts through three paths and
all three carry real signal.** You were right that the original result was confounded.

What survives for the Ne position, and it is the new thing here: **route 1 shows Ne alone**, at
identical K, identical census N and identical N·u, producing a monotone dose–response with
**r = −0.998** across a 7× range of structure, with erosion concentrated at late ages exactly as
the drift barrier predicts. No comparative dataset can isolate that — it needs an experiment.

Two things worth knowing about the comparison:

- **The ranges are not commensurable.** We moved F_ST 7.5× and µ 8×. Which matters more *in
  nature* depends on which varies more between real populations — and fragmentation varies
  enormously between killifish pools, while germline µ is comparatively conserved within a
  species. Route 2 having the larger per-knob effect does not make it the ecologically dominant
  path.
- **The late/early ratio is a second discriminator.** Supply is the most uniform (3.8×) because
  extra mutations arrive at every age; extrinsic mortality is the most late-concentrated (10.5×)
  because it acts directly on the selection gradient. That gives a way to tell mechanisms apart
  in real data independently of effect size.

---

## 4. Second experiment — annual killifish and the water window

![killifish](../../runs/killifish_windows.png)

Dario's scenario: killifish live about as long as their pool holds water, while time to maturity
is conserved. Explanation to test — a long-lived ancestor colonises faster-drying pools, and
mutations affecting survival *beyond* the water window accumulate because selection cannot see
past it.

Same ancestor, plus a periodic total dry-down that kills every fish but spares the diapausing
egg bank. Windows of 12/18/24/30 steps (2–5× age at maturity), each run with two generation
structures.

**With synchronous annual cohorts, evolved lifespan tracks the window almost one-for-one:**

| window | 12 | 18 | 24 | 30 |
|---|--:|--:|--:|--:|
| evolved lifespan | 13.0 | 17.2 | 20.3 | 21.6 |

saturating once the window exceeds what the ancestor had anyway (20.9). The survival curve shows
a **knee at exactly the window** in every arm — flat before it, collapsed after.

**With overlapping generations it does not.** Those arms degrade early regardless of the window
(lifespans 12.6 / 12.0 / 13.8 / 14.7 — barely moving). The reason is survivorship shape: with a
synchronous cohort *everyone* reaches the window, so the horizon is a hard step; with continuous
hatching an individual born late reaches only a young age, so selection weakens gradually from
birth.

**So the egg bank is not incidental — it is the mechanism that makes lifespan track water
sharply.** Comparative prediction: non-annual congeners in the same pools should show shorter,
less tightly coupled lifespans.

---

## 5. Where this sits in the literature

Worth knowing before writing anything up: **Ne → lifespan is not untouched ground.**

- **Lohr, David & Haag (2014)** *Evolution* — empirical, *Daphnia magna* across pond sizes; reduced
  lifespan and accelerated ageing in small populations, attributed to drift rather than adaptation.
  Five years *before* Cui et al. 2019.
- **Lehtonen (2020)** *Evolution Letters* — the analytical drift-barrier theory, giving maximum
  lifespan ≈ b + ln(Ne)/(σf).
- **Aubier & Galipaud (2024)** *Evolution Letters* — extends it, and proposes a *competing*
  mechanism where senescence ratchets earlier without Ne changing at all.

What is not in that literature: an experimental decomposition holding K, N and N·u fixed. That is
what route 1 provides, and it is also a direct test of Lehtonen's ln(Ne) law by an independent
method.

---

## 6. Caveats

- Three seeds, one genetic architecture, `AGE_LIMIT = 30`.
- Seeds 2 and 3 branched at 2.5% / 2.9% of the neutral gap remaining rather than <1%. Within-seed
  contrasts are unaffected (all arms of a seed share one ancestor); seed 1 is clean as a reference.
- The overlapping killifish arms vary in population size across windows (979–1698), so their
  effect sizes carry an Ne confound. The annual arms do not — N is 1509–1570 across all windows,
  a 4% spread against an 8.6-unit lifespan effect.
- Route 3's arm had to be run twice: the first version silently applied no mortality at all
  (`STARVATION_PENALTY` was 0, which nulls the penalty entirely). Worth knowing as a general
  warning — it exited cleanly and produced a plausible three-seed result.

---

## 7. What's open — and it's yours

The reproductive axis. Every result above is **survival**; reproduction was locked
(`G_repr_evolvable = False`) throughout.

That axis is where the literature is thinnest. Lehtonen and Aubier & Galipaud both model
*mortality*; Lohr measured *lifespan*; Cui and Willemsen measured lifespan and mutation load.
**None of them treat fecundity.** And Hamilton's selection gradients differ between survival and
fertility — the two kernels have different age weightings — so there is no reason they should
erode at the same rate under drift. Whether they do is open, testable, and needs a system where
the genetic architecture is visible.

Practical notes for that experiment: `G_repr_evolvable` and `G_repr_agespecific` are *structural*
parameters, so it needs its own burn-in and cannot branch off the current ancestor. Turning
reproduction on also needs lo/hi pre-compensation (`G_repr_lo: 0, G_repr_hi: 0.7071`) — see
`HANDOFF.md`. The three-route design transfers to it directly.
