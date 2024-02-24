"""Modulator of phenotypic effects

Transforms interpreter values of loci into phenotypic values of traits.
When no PHENOMAP_SPECS are given, the interpreter values of loci are phenotypic values of traits they are encoding.
When PHENOMAP_SPECS are given, interpreter values of some loci affect the phenotypic values not only of traits they are
    directly encoding but also of other traits.

Phenotypic values can loosely be understood as levels of final quantitative traits, and some loci affecting
    multiple traits can be understood as pleiotropy.

Example of PHENOMAP_SPECS: 
    [
    ["neut", 1, "surv", '12', '0.0015022406369215543'], 
    ["neut", 1, "surv", '17', '-0.0015022406369215543'],
    ["neut", 2, "surv", '1', '0.0004318344017753981'], 
    ["neut", 2, "surv", '10', '-0.0004318344017753981'], 
    ["neut", 3, "surv", '2', '0.007525611621651378'],
    ["neut", 3, "surv", '20', '-0.007525611621651378'], 
    ["neut", 4, "surv", '12', '0.0005129979580597647'],
    ["neut", 4, "surv", '29', '-0.0005129979580597647'], 
    ["neut", 5, "surv", '7', '0.00016150685474952513'],
    ["neut", 5, "surv", '18', '-0.00016150685474952513'], 
    ["neut", 6, "surv", '1', '0.0025197159199181027'],
    ["neut", 6, "surv", '4', '-0.0025197159199181027'], 
    ["neut", 7, "surv", '10', '0.0006980947369962286'],
    ["neut", 7, "surv", '10', '-0.0006980947369962286'], 
    ["neut", 8, "surv", '33', '0.0007719899275531902'],
    ["neut", 8, "surv", '46', '-0.0007719899275531902'], 
    ["neut", 9, "surv", '7', '4.159881874718622e-05'],
    ["neut", 9, "surv", '49', '-4.159881874718622e-05'], 
    ["neut", 10, "surv", '32', '0.001069693141890309'],
    ["neut", 10, "surv", '40', '-0.001069693141890309'], 
    ["neut", 11, "surv", '7', '0.0014580417800865069'],
    ["neut", 11, "surv", '15', '-0.0014580417800865069'],
    ]

"""

import numpy as np
import pandas as pd
from aegis import cnf
from aegis.modules.genetics.gstruc import traits as traits_
from aegis.modules.genetics.gstruc import length as length_


def _by_dummy(probs):
    """A dummy method for calc function"""
    return probs


def _by_dot(probs):
    """A vectorized method for calc function

    Use when map_ is dense.
    """
    return clip(probs.dot(map_))


def _by_loop(probs):
    """A naive method for calc function

    Use when map_ is sparse.
    Note that it modifies probs in place.
    """

    # List phenotypic differences caused by loci1, scaled by the given phenomap weights
    diffs = probs[:, _by_loop_loc1] * _by_loop_weights

    # Override baseline weights
    probs[:, _by_loop_loc_self] = 0

    # Add back the phenotypic differences caused by loci1 to loci2
    df = pd.DataFrame(diffs.T).groupby(_by_loop_loc2).sum()
    loc2 = tuple(df.index)
    probs[:, loc2] += df.to_numpy().T

    return clip(probs)


# Fully static methods


def clip(array):
    """Faster version of np.clip(?, 0, 1)"""
    array[array > 1] = 1
    array[array < 0] = 0
    return array


def decode_scope(scope):
    if "," in scope:
        loci = scope.split(",")
    elif "-" in scope:
        from_, to_ = scope.split("-")
        loci = list(range(int(from_), int(to_) + 1))
    else:
        loci = [scope]
    return np.array(loci).astype(int)


def decode_pattern(pattern, n):
    decoded = pattern.split(",")
    first = float(decoded.pop(0))
    last = float(decoded[0]) if decoded else first

    if n == 1 and last != first:
        raise ValueError(f"Pattern '{pattern}' contains two values but there is only one target locus")

    return np.linspace(first, last, n)


def unfold_specs():
    for trait1, scope1, trait2, scope2, pattern2 in cnf.PHENOMAP_SPECS:
        assert (trait1 is None and scope1 is None) or (trait1 is not None and scope2 is not None)

        # If no scope given, whole trait is affected
        if scope2 is None:
            scope2 = f"{traits_[trait2].start + 1}-{traits_[trait2].end}"
            # Note that PHENOMAP_SPECS scope is interpreted as a 1-indexed inclusive interval

        pos2 = traits_[trait2].start
        loci2 = decode_scope(scope2) + pos2 - 1  # -1 because the PHENOMAP_SPECS is 1-indexed
        weights = decode_pattern(pattern2, len(loci2))

        if trait1 is None:
            loci1 = loci2
        else:
            pos1 = traits_[trait1].start
            loci1 = [scope1 + pos1 - 1] * len(loci2)

        for locus1, locus2, weight in zip(loci1, loci2, weights):
            yield locus1, locus2, weight


if cnf.PHENOMAP_SPECS == []:
    map_ = None
    call = _by_dummy
else:
    trios = list(unfold_specs())

    map_ = np.diag([1.0] * length_)
    for locus1, locus2, weight in trios:
        map_[locus1, locus2] = weight

    call = {
        "by_dot": _by_dot,
        "by_loop": _by_loop,
    }[cnf.PHENOMAP_METHOD]

    # Variables for Phenomap._by_loop
    _ = np.array(list(zip(*trios)))
    _by_loop_loc1 = _[0].astype(int)
    _by_loop_loc2 = _[1].astype(int)
    _by_loop_weights = _[2]
    _by_loop_loc_self = _by_loop_loc1[
        _by_loop_loc1 == _by_loop_loc2
    ]  # Loci that affect themselves; i.e. change the baseline weight from 1 to something else
