# `python3 -m pytest tests/test_popgen_theta_pi.py --log-cli-level=DEBUG`

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
    "theta_pi_valid_input,expected",
    [
        ((example1, 4), 7.66667),
        ((example2, 2), 4.0),
        ((example3, 2), 3.0),
        ((example4, 2), 0.0),
        ((example5, 1), None),
        ((example6, 1), None),
        ((example1, None, "asexual", True), 7.66667),
        ((example1, None, "sexual", True), 5.5),
        ((example1, 8, "sexual"), 5.5),
    ],
)
def test_theta_pi_valid_input(theta_pi_valid_input, expected):
    tpi = popgenstats.theta_pi(*theta_pi_valid_input)
    assert pytest.approx(tpi, 0.0001) == expected
