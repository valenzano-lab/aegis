import pytest
import aegis


@pytest.mark.parametrize(
    "config_file",
    [
        "RANDOM_SEED_",
        "OVERSHOOT_EVENT_cliff",
        "OVERSHOOT_EVENT_treadmill_boomer",
        "OVERSHOOT_EVENT_treadmill_zoomer",
        "OVERSHOOT_EVENT_treadmill_random",
        "STAGES_PER_SEASON",
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
        "STAGES_PER_SIMULATION_",
        "ENVIRONMENT_CHANGE_RATE",
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
