import pytest
import aegis


@pytest.mark.parametrize(
    "config_file",
    [
        "RANDOM_SEED",
        "STARVATION_RESPONSE_cliff",
        "STARVATION_RESPONSE_treadmill_boomer",
        "STARVATION_RESPONSE_treadmill_zoomer",
        "STARVATION_RESPONSE_treadmill_random",
        "MAX_LIFESPAN",
        "MATURATION_AGE",
        "BITS_PER_LOCUS",
        "HEADSUP_0",
        "HEADSUP_plus",
        "REPRODUCTION_MODE_sexual",
        "REPRODUCTION_MODE_asexual_diploid",
        "RECOMBINATION_RATE",
        "MUTATION_RATIO",
        "PHENOMAP_SPECS",
        "STAGES_PER_SIMULATION",
        "FLIPMAP_CHANGE_RATE",
    ],
)
def test_execution(config_file):
    """Test if execution is successful with some other non-default parameter values."""
    aegis.main(
        arg_dict={
            "custom_config_path": f"tests/execution/{config_file}.yml",
            "pickle_path": "",
        }
    )
