# `python3 -m pytest tests/test_popgen_get_theta.py --log-cli-level=DEBUG`

import pytest

from aegis.classes import popgenstats


@pytest.mark.parametrize(
    "get_theta_valid_input,expected",
    [
        (("asexual", 20000, 0.0001), 4.0),
        (("sexual", 20000, 0.000001), 0.08),
        (("sexual", 100, 0.0001), 0.04),
        (("asexual_diploid", 1000, 5e-7), 0.002),
    ],
)
def test_get_theta_valid_input(get_theta_valid_input, expected):
    theta = popgenstats.get_theta(*get_theta_valid_input)
    assert pytest.approx(theta, 0.0001) == expected
