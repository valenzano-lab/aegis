import pytest
import numpy as np
from aegis.modules.reproducer import Reproducer

reproducer = Reproducer(0.5, 0.1, "sexual")


@pytest.mark.parametrize(
    "genomes,muta_prob,random_probabilities,expected",
    [
        (
            [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
            [0.42],
            [[[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]]],
            [[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]],
        ),
        (
            [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            [0.42],
            [[[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]]],
            [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
        ),
        (
            [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            [0.042],
            [[[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]]],
            [[[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]],
        ),
    ],
)
def test_mutate(genomes, muta_prob, random_probabilities, expected):
    result = reproducer._mutate(
        np.array(genomes), np.array(muta_prob), np.array(random_probabilities)
    )
    result = result.astype(int)
    assert np.allclose(result, np.array(expected)), f"{result} {expected}"


@pytest.mark.parametrize(
    "genomes,order,expected",
    [
        (
            [
                [[[0, 0, 0, 0]], [[1, 1, 1, 1]]],
                [[[0, 0, 1, 1]], [[1, 1, 0, 0]]],
                [[[1, 0, 0, 1]], [[0, 1, 1, 0]]],
            ],
            [0, 1, 1, 2, 2, 0],
            [
                [[[0, 0, 0, 0]], [[1, 1, 0, 0]]],
                [[[0, 0, 1, 1]], [[0, 1, 1, 0]]],
                [[[1, 0, 0, 1]], [[1, 1, 1, 1]]],
            ],
        ),
        (
            [
                [[[0, 0, 0, 0]], [[1, 1, 1, 1]]],
                [[[0, 0, 1, 1]], [[1, 1, 0, 0]]],
                [[[1, 0, 0, 1]], [[0, 1, 1, 0]]],
            ],
            [1, 0, 0, 2, 2, 1],
            [
                [[[0, 0, 1, 1]], [[1, 1, 1, 1]]],
                [[[0, 0, 0, 0]], [[0, 1, 1, 0]]],
                [[[1, 0, 0, 1]], [[1, 1, 0, 0]]],
            ],
        ),
    ],
)
def test_assort(genomes, order, expected):
    result, _ = reproducer._assort(np.array(genomes), np.array(order))
    assert np.array_equal(result, np.array(expected)), f"{result} {expected}"


@pytest.mark.parametrize(
    "order,expected",
    [
        # len(selfed) > 1, thus roll one
        ([0, 0, 1, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1, 2, 2], [2, 0, 0, 1, 1, 2]),
        ([0, 0, 1, 1, 2, 2, 3, 3], [3, 0, 0, 1, 1, 2, 2, 3]),
        # len(selfed) == 1
        ([0, 1, 1, 0, 2, 2], [0, 1, 2, 0, 1, 2]),
        ([0, 1, 1, 2, 3, 3, 0, 1], [0, 1, 3, 2, 1, 3, 0, 1]),
        ([0, 0, 1, 2, 2, 1], [1, 0, 0, 2, 2, 1]),
    ],
)
def test_get_order(order, expected):
    result = reproducer._get_order(order=np.array(order))
    assert np.array_equal(result, np.array(expected)), f"{result} {expected}"

# TODO _recombine