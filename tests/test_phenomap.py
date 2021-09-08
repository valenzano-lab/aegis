import pytest
import numpy as np
from aegis.modules.phenomap import Phenomap


def test_dummy():
    phenomap = Phenomap([], None)
    assert phenomap.dummy


@pytest.mark.parametrize(
    "PHENOMAP_SPECS,gstruc_length,expected",
    [
        # nullify effect
        ([[0, 0, 0]], 2, [[0, 0], [0, 1]]),
        # multiply effect
        ([[0, 0, 2]], 2, [[2, 0], [0, 1]]),
        ([[0, 0, 0.42]], 2, [[0.42, 0], [0, 1]]),
        # pleiotropy
        ([[0, 1, 0.42]], 2, [[1, 0.42], [0, 1]]),
        ([[0, 1, -0.42]], 2, [[1, -0.42], [0, 1]]),
        # multiple effects
        ([[0, 1, 0.42], [0, 2, -0.42]], 3, [[1, 0.42, -0.42], [0, 1, 0], [0, 0, 1]]),
        ([[0, 1, 0.42], [1, 2, -0.42]], 3, [[1, 0.42, 0], [0, 1, -0.42], [0, 0, 1]]),
    ],
)
def test_map_(PHENOMAP_SPECS, gstruc_length, expected):
    phenomap = Phenomap(PHENOMAP_SPECS, gstruc_length)
    assert np.array_equal(phenomap.map_, np.array(expected))


phenomap = Phenomap([[0, 1, 0.42], [1, 2, -0.42]], 3)


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([1, 1, 1], [1, 1, 0.58]),
        ([0.1, 0.1, 0.1], [0.1, 0.142, 0.058]),
        ([1, 0.1, 0.1], [1, 0.52, 0.058]),
        ([0.1, 1, 0.1], [0.1, 1, 0]),
    ],
)
def test_call(probs, expected):
    result = phenomap(np.array(probs))
    assert np.allclose(result, np.array(expected)), f"{result} {expected}"
