# python3 -m pytest tests/test_phenomap.py --log-cli-level=DEBUG

import pytest
import numpy as np
from aegis.modules.phenomap import Phenomap


def test_dummy():
    phenomap = Phenomap([], None)
    assert phenomap.dummy


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("0", [0]),
        ("1", [1]),
        ("1,2", [1, 2]),
        ("0-0", [0]),
        ("0-2", [0, 1, 2]),
        ("1-2", [1, 2]),
        ("2-7", [2, 3, 4, 5, 6, 7]),
    ],
)
def test_decode_scope(scope, expected):
    result = Phenomap.decode_scope(scope)
    assert np.allclose(result, np.array(expected)), f"{result} {expected}"


@pytest.mark.parametrize(
    "pattern,n,expected",
    [
        # Test 1 locus, positive
        ("0", 1, [0]),
        ("0.1", 1, [0.1]),
        ("1", 1, [1]),
        ("10", 1, [10]),
        # Test 2 loci, positive
        ("0,1", 2, [0, 1]),
        ("0.1,1", 2, [0.1, 1]),
        ("0.5,1", 2, [0.5, 1]),
        ("0.5,7", 2, [0.5, 7]),
        # Test 3 loci, positive
        ("0,1", 3, [0, 0.5, 1]),
        ("0.1,1", 3, [0.1, 0.55, 1]),
        ("0.5,1", 3, [0.5, 0.75, 1]),
        ("0.5,1.5", 3, [0.5, 1, 1.5]),
        # Test 1 locus, negative
        ("-0.1", 1, [-0.1]),
        ("-3", 1, [-3]),
        # Test 2 loci, negative
        ("-0.1,1", 2, [-0.1, 1]),
        ("0.1,-1", 2, [0.1, -1]),
        ("-0.1,-1", 2, [-0.1, -1]),
        # Test 3 loci, negative
        ("0.5,-1", 3, [0.5, -0.25, -1]),
        ("-0.5,1", 3, [-0.5, 0.25, 1]),
        ("-0.5,-1", 3, [-0.5, -0.75, -1]),
    ],
)
def test_decode_pattern(pattern, n, expected):
    result = Phenomap.decode_pattern(pattern, n)
    assert np.allclose(result, np.array(expected)), f"{result} {expected}"


@pytest.mark.parametrize(
    "pos1, scope1, pos2, scope2, pattern2, expected",
    [
        # Test pos1
        (0, 0, 314, "0", "0.5", [(0, 314, 0.5)]),
        (42, 0, 314, "0", "0.5", [(42, 314, 0.5)]),
        # Test scope1
        (0, 141, 314, "0", "0.5", [(141, 314, 0.5)]),
        (42, 141, 314, "0", "0.5", [(183, 314, 0.5)]),
        # Test pos2
        (0, 0, 173, "0", "0.5", [(0, 173, 0.5)]),
        (42, 0, 224, "0", "0.5", [(42, 224, 0.5)]),
        # Test scope2
        (0, 0, 173, "1", "0.5", [(0, 174, 0.5)]),
        (42, 0, 224, "1", "0.5", [(42, 225, 0.5)]),
        (0, 0, 173, "7", "0.5", [(0, 180, 0.5)]),
        (42, 0, 224, "7", "0.5", [(42, 231, 0.5)]),
        (0, 0, 173, "0-0", "0.5", [(0, 173, 0.5)]),
        (42, 0, 224, "1-1", "0.5", [(42, 225, 0.5)]),
        (0, 0, 173, "0-1", "0.5", [(0, 173, 0.5), (0, 174, 0.5)]),
        (0, 0, 173, "0-1", "0.5,1", [(0, 173, 0.5), (0, 174, 1)]),
        (0, 0, 173, "0-1", "0.2,0.73", [(0, 173, 0.2), (0, 174, 0.73)]),
        (
            0,
            0,
            173,
            "5-7",
            "0.2,0.8",
            [(0, 178, 0.2), (0, 179, 0.5), (0, 180, 0.8)],
        ),
    ],
)
def test_decode_spec(pos1, scope1, pos2, scope2, pattern2, expected):
    result = list(Phenomap.decode_spec(pos1, scope1, pos2, scope2, pattern2))
    assert np.allclose(result, np.array(expected)), f"{result} {expected}"
