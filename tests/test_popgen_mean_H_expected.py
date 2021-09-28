# `python3 -m pytest tests/test_popgen_mean_h_expected.py --log-cli-level=DEBUG`

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
    "mean_h_expected_valid_input,expected",
    [
        (
            (example1, "asexual_diploid"),
            0.497396,
        ),
        ((example2, "sexual"), 0.375),
        ((example3, "asexual_diploid"), 0.5),
        ((example4, "sexual"), 0.0),
        ((example5, "asexual"), None),
        ((example6, "asexual"), None),
    ],
)
def test_mean_h_expected_valid_input(mean_h_expected_valid_input, expected):
    h = popgenstats.mean_h_expected(*mean_h_expected_valid_input)
    assert pytest.approx(h, 0.0001) == expected
