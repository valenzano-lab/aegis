import numpy as np
from aegis.pan import rng


def _get_order(n_gametes=None, order=None):
    """Return pairings of gametes from different parents."""
    # Extract parent indices twice, and shuffle
    if order is None:
        order = np.repeat(np.arange(n_gametes), 2)
        rng.shuffle(order)

    # Check for selfing (selfing when pair contains equal parent indices)
    selfed = (order[::2] == order[1::2]).nonzero()[0] * 2

    if len(selfed) == 1:
        # If one parent index pair is selfed,
        #   swap first selfed chromatid with the first chromatid of the previous or next pair
        offset = 2 if selfed[0] == 0 else -2
        order[selfed], order[selfed + offset] = (
            order[selfed + offset],
            order[selfed],
        )
    elif len(selfed) > 1:
        # If multiple parent index pairs are selfed, shift first chromatid of selfed pairs
        order[selfed] = order[np.roll(selfed, 1)]

    return order


def do(genomes, order=None):
    """Return assorted chromatids."""

    if order is None:
        order = _get_order(n_gametes=len(genomes))

    # Extract gametes
    gametes = genomes.get(individuals=order)

    # Unify gametes
    children = np.empty(genomes.get_array().shape, dtype=np.bool_)
    children[:, 0] = gametes[::2, 0]  # 1st chromatid from 1st parent
    children[:, 1] = gametes[1::2, 1]  # 2nd chromatid from 2nd parent

    return children, order
