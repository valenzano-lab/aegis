"""selectable_via_surv: a genetic trait whose phenotype adds a selection-derived
correction to the surv phenotype.

Like any other genetic trait, selectable_via_surv has its own genome loci, is
heritable, mutable, and (under the composite architecture) evolves under whichever
interpreter/lo/hi/initgeno the user configures for it (G_selectable_via_surv_*).
Its phenotype is computed and exposed exactly like surv/repr -- it can be recorded
and inspected on its own.

On top of that, once per phenotype computation (see Architect.__call__), its
phenotype value is passed through one of two selection functions and the result is
added to the surv phenotype, then surv is hard-clipped to [0, 1]:

- directional_selection: a step function. Trait value >= cutoff adds a constant
  benefit; below cutoff adds a constant (typically negative) penalty.
- stabilizing_selection: a Gaussian centered on a target trait value, up to N
  standard deviations from the mean; beyond that, a constant (typically negative)
  penalty applies instead of the Gaussian tail.
"""

import numpy as np

from aegis_sim import parameterization
from aegis_sim.parameterization import parametermanager


def _get_trait():
    return parameterization.traits.get("selectable_via_surv")


def is_active():
    trait = _get_trait()
    return trait is not None and trait.evolvable


def validate():
    """Check config consistency. Called once at Architect init time so bad config
    fails fast instead of surfacing mid-simulation."""

    if not is_active():
        return

    trait = _get_trait()
    surv = parameterization.traits["surv"]

    if trait.length not in (1, surv.length):
        raise ValueError(
            f"selectable_via_surv has {trait.length} loci but surv has {surv.length}; "
            "G_selectable_via_surv_agespecific must produce either 1 value "
            "(broadcast to all ages of surv) or exactly match surv's length."
        )


def apply(phenotypes):
    """Add the selectable_via_surv-derived correction to the surv phenotype, in place."""

    if not is_active():
        return phenotypes

    trait = _get_trait()
    surv = parameterization.traits["surv"]
    p = parametermanager.parameters

    value = phenotypes.array[:, trait.slice]  # already lo/hi-scaled and smoothed

    mode = p.G_selectable_via_surv_mode
    if mode == "directional_selection":
        delta = _directional_delta(value, p)
    elif mode == "stabilizing_selection":
        delta = _stabilizing_delta(value, p)
    else:
        raise ValueError(f"Unknown G_selectable_via_surv_mode: {mode!r}")

    delta = _broadcast_to_surv_shape(delta, surv)

    phenotypes.array[:, surv.slice] += delta
    phenotypes.array[:, surv.slice] = np.clip(phenotypes.array[:, surv.slice], 0, 1)

    return phenotypes


def _directional_delta(value, p):
    cutoff = p.G_selectable_via_surv_directional_sel_cutoff_value
    benefit = p.G_selectable_via_surv_directional_sel_constant_benefit
    penalty = p.G_selectable_via_surv_directional_sel_constant_penalty
    return np.where(value >= cutoff, benefit, penalty)


def _stabilizing_delta(value, p):
    mean = p.G_selectable_via_surv_stabilizing_sel_mean
    sd = p.G_selectable_via_surv_stabilizing_sel_sd
    max_benefit = p.G_selectable_via_surv_stabilizing_sel_max_benefit
    const_penalty = p.G_selectable_via_surv_stabilizing_sel_const_penalty
    n_sd = p.G_selectable_via_surv_stabilizing_sel_const_penalty_beyond_sd

    z = (value - mean) / sd
    gaussian = max_benefit * np.exp(-0.5 * z**2)
    return np.where(np.abs(z) <= n_sd, gaussian, const_penalty)


def _broadcast_to_surv_shape(delta, surv):
    if delta.shape[1] == surv.length:
        return delta
    return np.repeat(delta, surv.length, axis=1)
