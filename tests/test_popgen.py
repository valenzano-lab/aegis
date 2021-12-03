# python3 -m pytest tests/test_popgen.py --log-cli-level=DEBUG

import pytest
import numpy as np
from aegis.modules import popgenstats
from aegis.panconfiguration import pan

pan.POPGENSTATS_SAMPLE_SIZE_ = 0

def get_popgenstats(genomes, ploidy=1, mutation_rates=0):
    obj = popgenstats.PopgenStats()
    genomes4D = obj.make_4D(genomes, ploidy)
    restaggered = obj.make_3D(genomes4D)
    assert np.array_equal(restaggered, genomes)  # Check that conversion is correct
    obj.record_pop_size_history(genomes4D)
    obj.calc(genomes4D, mutation_rates)
    return obj
    # TODO fix naming


# TODO rewrite genomes as 4D
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


example7 = np.array(
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


example8 = np.array(
    [
        [
            [0],
        ],
        [
            [0],
        ],
    ]
)

example9 = np.array(
    [
        [
            [1],
        ],
        [
            [1],
        ],
    ]
)


example10 = np.array(
    [
        [[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0]],
        [[0, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0]],
    ]
)
example11 = np.array([[[1], [1], [1], [0]], [[0], [0], [1], [0]]])

example12 = np.array(
    [
        [
            [1],
        ],
        [
            [0],
        ],
    ]
)


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (
            example1,
            2,
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
        (example2, 2, np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 2, 0]) / 2),
        (example3, 2, np.array([1, 0, 1, 0, 1, 1]) / 2),
        (example4, 2, np.array([0, 0, 2]) / 2),
        (example5, 1, np.array([0])),
        (example6, 1, np.array([1])),
    ],
)
def test_genotype_frequencies_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).genotype_frequencies
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (example1, 2, 0.497396),
        (example2, 2, 0.375),
        (example3, 2, 0.5),
        (example4, 2, 0.0),
        (example5, 1, None),
        (example6, 1, None),
    ],
)
def test_mean_h_expected_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).mean_h_expected
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (
            example1,
            2,
            np.array(
                [
                    0.5,
                    0.65625,
                    0.46875,
                    0.46875,
                    0.46875,
                    0.65625,
                    0.65625,
                    0.375,
                    0.21875,
                    0.46875,
                    0.65625,
                    0.375,
                ]
            ),
        ),
        (example2, 2, np.array([0.5, 0.5, 0.5, 0])),
        (example3, 2, np.array([0.5, 0.5])),
        (example4, 2, np.array([0])),
        (example5, 1, None),
        (example6, 1, None),
    ],
)
def test_mean_h_per_bit_expected_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).mean_h_per_bit_expected
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (
            example1,
            2,
            np.array([4, 3, 3, 3, 3, 3, 3, 6, 7, 3, 3, 6]) / 8,
        ),
        (example2, 2, np.array([1, 0, 1, 2]) / 2),
        (example3, 2, np.array([0, 1]) / 2),
        (example4, 2, np.array([0]) / 2),
        (example5, 1, None),
        (example6, 1, None),
    ],
)
def test_mean_h_per_bit_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).mean_h_per_bit
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (example1, 2, np.array([0.41667, 0.375, 0.66667, 0.5])),
        (example2, 2, np.array([1, 0, 1, 2]) / 2),
        (example3, 2, np.array([0, 1]) / 2),
        (example4, 2, np.array([0]) / 2),
        (example5, 1, None),
        (example6, 1, None),
    ],
)
def test_mean_h_per_locus_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).mean_h_per_locus
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (
            example1,
            2,
            np.mean(np.array([7, 6, 8, 7, 6, 7, 3, 3]) / 12),
        ),
        (example2, 2, 0.5),
        (example3, 2, 0.25),
        (example4, 2, 0),
        (example5, 1, None),
        (example6, 1, None),
    ],
)
def test_mean_h_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).mean_h
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (example1, 1, 17),
        (example2, 1, 4),
        (example3, 1, 3),
        (example4, 1, 0),
        (example5, 1, 0),
        (example6, 1, 0),
        (example1, 2, 12),
    ],
)
def test_segregating_sites_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).segregating_sites
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,sample_size,ploidy,sample_provided,expected",
    [
        (example7, None, 1, False, 1.72761),
        (example7, 4, 1, False, 1.72761),
        (example2, 2, 1, False, None),
        (example3, 2, 1, False, None),
        (example4, 2, 1, False, None),
        (example5, 1, 1, False, None),
        (example6, 1, 1, False, None),
        (example7, None, 1, True, 1.72761),
        (example7, None, 2, True, 0.94789),
        (example7, 8, 2, False, 0.94789),
    ],
)
def test_tajimas_d_valid_input(genomes, sample_size, ploidy, sample_provided, expected):
    sample_size = 0 if sample_size is None else sample_size
    pan.POPGENSTATS_SAMPLE_SIZE_ = sample_size
    calculated = get_popgenstats(genomes=genomes, ploidy=ploidy).tajimas_d
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,sample_size,ploidy,sample_provided,expected",
    [
        (example7, 4, 1, False, 7.66667),
        (example2, 2, 1, False, 4.0),
        (example3, 2, 1, False, 3.0),
        (example4, 2, 1, False, 0.0),
        (example5, 1, 1, False, None),
        (example6, 1, 1, False, None),
        (example7, None, 1, True, 7.66667),
        (example7, None, 2, True, 5.5),
        (example7, 8, 2, False, 5.5),
    ],
)
def test_theta_pi_valid_input(genomes, sample_size, ploidy, sample_provided, expected):
    sample_size = 0 if sample_size is None else sample_size
    pan.POPGENSTATS_SAMPLE_SIZE_ = sample_size
    calculated = get_popgenstats(genomes=genomes, ploidy=ploidy).theta_pi
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,sample_size,ploidy,sample_provided,expected",
    [
        (example1, 8, 1, False, 6.55647),
        (example2, 2, 1, False, 4.0),
        (example3, 2, 1, False, 3.0),
        (example4, 2, 1, False, 0.0),
        (example5, 1, 1, False, None),
        (example6, 1, 1, False, None),
        (example1, None, 1, True, 6.55647),
        (example1, None, 2, True, 3.61639),
    ],
)
def test_theta_w_valid_input(genomes, sample_size, ploidy, sample_provided, expected):
    sample_size = 0 if sample_size is None else sample_size
    pan.POPGENSTATS_SAMPLE_SIZE_ = sample_size
    calculated = get_popgenstats(genomes=genomes, ploidy=ploidy).theta_w
    assert pytest.approx(calculated, 0.0001) == expected


@pytest.mark.parametrize(
    "genomes,ploidy,expected",
    [
        (
            example1,
            1,
            np.array(
                [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
            ),
        ),
        (example2, 1, np.array([0, 1, 0, 0, 1, 0, 1, 0])),
        (example3, 1, np.array([0, 0, 1, 0])),
        (example4, 1, np.array([1, 1])),
        (example8, 1, np.array([0])),
        (example9, 1, np.array([1])),
    ],
)
def test_reference_genome_valid_input(genomes, ploidy, expected):
    calculated = get_popgenstats(genomes, ploidy).reference_genome_gsample
    assert pytest.approx(calculated, 0.0001) == expected


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
        (example10, np.array([1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 0, 0]) / 2),
        (example11, np.array([1, 1, 2, 0]) / 2),
        (example12, np.array([0.5])),
        (example5, np.array([0])),
        (example6, np.array([1])),
    ],
)
def test_allele_frequencies_valid_input(allele_frequencies_valid_input, expected):
    calculated = get_popgenstats(allele_frequencies_valid_input).allele_frequencies
    assert pytest.approx(calculated, 0.0001) == expected
