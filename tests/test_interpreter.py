import pytest
import numpy as np
from aegis.modules.interpreter import Interpreter


e = Interpreter.exp_base
be = Interpreter.binary_exp_base


@pytest.mark.parametrize(
    "BITS_PER_LOCUS,loci,expected",
    [
        # 1 individual 3 loci
        (4, [[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]], [[0, 1 / 15, 2 / 15]]),
        # 2 individuals 1 locus
        (4, [[[0, 1, 0, 0]], [[1, 1, 1, 0]]], [[4 / 15], [14 / 15]]),
        # 1 individual 1 locus
        (4, [[[1, 0, 0, 0]]], 8 / 15),
        (4, [[[1, 0, 0, 1]]], 9 / 15),
        (4, [[[1, 0, 1, 0]]], 10 / 15),
        (4, [[[1, 1, 0, 0]]], 12 / 15),
        (4, [[[1, 1, 1, 1]]], 1),
        (5, [[[0, 0, 0, 0, 0]]], 0),
        (5, [[[0, 0, 0, 0, 1]]], 1 / 31),
        (5, [[[0, 0, 0, 1, 0]]], 2 / 31),
        (5, [[[0, 0, 1, 0, 0]]], 4 / 31),
        (5, [[[0, 1, 0, 0, 0]]], 8 / 31),
        (5, [[[1, 0, 0, 0, 0]]], 16 / 31),
        (5, [[[1, 0, 0, 0, 1]]], 17 / 31),
        (5, [[[1, 1, 1, 1, 1]]], 1),
    ],
)
def test_binary(BITS_PER_LOCUS, loci, expected):
    interpreter = Interpreter(BITS_PER_LOCUS)
    result = interpreter._binary(np.array(loci))
    assert (expected == result).all()


@pytest.mark.parametrize(
    "BITS_PER_LOCUS,loci,expected",
    [
        # 1 individual 3 loci
        (4, [[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]]], [[0, 0, 1 / 7]]),
        # 2 individuals 1 locus
        (4, [[[0, 1, 0, 1]], [[1, 1, 0, 1]]], [[2 / 7], [6 / 7]]),
        # 1 individual 1 locus
        (4, [[[0, 0, 0, 0]]], 0),
        (4, [[[0, 1, 1, 0]]], 0),
        (4, [[[0, 1, 1, 1]]], 3 / 7),
        (4, [[[1, 0, 0, 0]]], 0),
        (4, [[[1, 0, 0, 1]]], 4 / 7),
    ],
)
def test_binary_switch(BITS_PER_LOCUS, loci, expected):
    interpreter = Interpreter(BITS_PER_LOCUS)
    result = interpreter._binary_switch(np.array(loci))
    assert (expected == result).all()


@pytest.mark.parametrize(
    "BITS_PER_LOCUS,loci,expected",
    [
        # 1 individual 3 loci
        (4, [[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]]], [[0, 1 / 4, 2 / 4]]),
        # 2 individuals 1 locus
        (4, [[[0, 1, 0, 1]], [[1, 1, 0, 1]]], [[2 / 4], [3 / 4]]),
        # 1 individual 1 locus
        (4, [[[0, 0, 0, 0]]], 0),
        (4, [[[0, 1, 1, 0]]], 2 / 4),
        (4, [[[0, 1, 1, 1]]], 3 / 4),
        (4, [[[1, 0, 0, 0]]], 1 / 4),
        (4, [[[1, 0, 0, 1]]], 2 / 4),
    ],
)
def test_uniform(BITS_PER_LOCUS, loci, expected):
    interpreter = Interpreter(BITS_PER_LOCUS)
    result = interpreter._uniform(np.array(loci))
    assert (expected == result).all()


@pytest.mark.parametrize(
    "BITS_PER_LOCUS,loci,expected",
    [
        # 1 individual 3 loci
        (
            4,
            [[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]]],
            [[e ** 4, e ** 3, e ** 2]],
        ),
        # 2 individuals 1 locus
        (4, [[[0, 1, 0, 1]], [[1, 1, 0, 1]]], [[e ** 2], [e ** 1]]),
        # 1 individual 1 locus
        (4, [[[0, 0, 0, 0]]], e ** 4),
        (4, [[[0, 1, 1, 0]]], e ** 2),
        (4, [[[0, 1, 1, 1]]], e ** 1),
        (4, [[[1, 0, 0, 0]]], e ** 3),
        (4, [[[1, 0, 0, 1]]], e ** 2),
    ],
)
def test_exp(BITS_PER_LOCUS, loci, expected):
    interpreter = Interpreter(BITS_PER_LOCUS)
    result = interpreter._exp(np.array(loci))
    assert (expected == result).all()


@pytest.mark.parametrize(
    "BITS_PER_LOCUS,loci,expected",
    [
        # 1 individual 3 loci
        (
            4,
            [[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]]],
            [[be ** 0, be ** (1 / 15), be ** (3 / 15)]],
        ),
        # 2 individuals 1 locus
        (4, [[[0, 1, 0, 1]], [[1, 1, 0, 1]]], [[be ** (5 / 15)], [be ** (13 / 15)]]),
        # 1 individual 1 locus
        (4, [[[0, 0, 0, 0]]], be ** (0 / 15)),
        (4, [[[0, 1, 1, 0]]], be ** (6 / 15)),
        (4, [[[0, 1, 1, 1]]], be ** (7 / 15)),
        (4, [[[1, 0, 0, 0]]], be ** (8 / 15)),
        (4, [[[1, 0, 0, 1]]], be ** (9 / 15)),
    ],
)
def test_binary_exp(BITS_PER_LOCUS, loci, expected):
    interpreter = Interpreter(BITS_PER_LOCUS)
    result = interpreter._binary_exp(np.array(loci))
    assert (expected == result).all()
