# `python3 -m pytest tests/test_popgen_allele_frequencies.py --log-cli-level=DEBUG`

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
    [
        [[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0]],
        [[0, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0]],
    ]
)

example3 = np.array([[[1], [1], [1], [0]], [[0], [0], [1], [0]]])

example4 = np.array(
    [
        [
            [1],
        ],
        [
            [0],
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
    "allele_frequencies_valid_input,expected",
    [
        (
            example1,
            np.array(
                [5, 7, 5, 2, 8, 5, 3, 0, 5, 8, 3, 6, 3, 6, 2, 8, 8, 1, 3, 0, 3, 6, 0, 6]
            )
            / 8,
        ),
        (example2, np.array([1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 0, 0]) / 2),
        (example3, np.array([1, 1, 2, 0]) / 2),
        (example4, np.array([0.5])),
        (example5, np.array([0])),
        (example6, np.array([1])),
    ],
)
def test_allele_frequencies_valid_input(allele_frequencies_valid_input, expected):
    allele_freq = popgenstats.allele_frequencies(allele_frequencies_valid_input)
    assert pytest.approx(allele_freq, 0.0001) == expected
