# `python3 -m pytest tests/test_popgen_genotype_frequencies.py --log-cli-level=DEBUG`

import pytest
import numpy as np

from aegis.classes import popgenstats


example1 = np.array(
    [
        [
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        [
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0, 1],
        ],
        [
            [1, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        [
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1],
        ],
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
    ]
)

example2 = np.array(
    [[[1, 1], [1, 1], [1, 0], [1, 0]], [[0, 1], [0, 0], [1, 1], [1, 0]]]
)

example3 = np.array(
    [
        [
            [1, 1],
            [1, 1],
        ],
        [
            [0, 0],
            [1, 0],
        ],
    ]
)

example4 = np.array(
    [
        [
            [1, 1],
        ],
        [
            [1, 1],
        ],
    ]
)

example5 = np.array(
    [
        [
            [0],
        ]
    ]
)

example6 = np.array(
    [
        [
            [1],
        ]
    ]
)


@pytest.mark.parametrize(
    "genotype_frequencies_valid_input,expected",
    [
        (
            (example1, "sexual"),
            np.array(
                [
                    0,
                    4,
                    4,
                    3,
                    3,
                    2,
                    0,
                    3,
                    5,
                    5,
                    3,
                    0,
                    0,
                    3,
                    5,
                    2,
                    3,
                    3,
                    2,
                    3,
                    3,
                    0,
                    6,
                    2,
                    0,
                    7,
                    1,
                    5,
                    3,
                    0,
                    2,
                    3,
                    3,
                    2,
                    6,
                    0,
                ]
            )
            / 8,
        ),
        ((example2, "sexual"), np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 2, 0]) / 2),
        ((example3, "sexual"), np.array([1, 0, 1, 0, 1, 1]) / 2),
        ((example4, "asexual_diploid"), np.array([0, 0, 2]) / 2),
        ((example5, "asexual"), np.array([0])),
        ((example6, "asexual"), np.array([1])),
    ],
)
def test_genotype_frequencies_valid_input(genotype_frequencies_valid_input, expected):
    genotype_freq = popgenstats.genotype_frequencies(*genotype_frequencies_valid_input)
    assert pytest.approx(genotype_freq, 0.0001) == expected
