import pytest
import logging
import yaml
from tests.functional.conftest import test_experiment_path
from aegis_sim import run

logging.basicConfig(level=logging.INFO)

path = test_experiment_path.parent / "test_sim.yml"


@pytest.mark.parametrize(
    "custom_input_params",
    [
        {
            # Basic
            "STEPS_PER_SIMULATION": 100,
            "LOGGING_RATE": 100,
            "SNAPSHOT_FINAL_COUNT": 3,
            # Tested
            "GENARCH_TYPE": "modifying",
            "MODIF_GENOME_SIZE": mgs,
            "BITS_PER_LOCUS": 1,
            "PHENOMAP": {f"AP, {mgs}": [["surv", "agespec", -0.1], ["surv", "agespec", 0.05]]},
            # "G_surv_initgeno": 0,
            # "G_repr_initgeno": 0,
        }
        for mgs in [2, 100, 999, 1000]  # TODO fails for 1; is that wrong?
    ],
)
def test_MODIF_GENOME_SIZE(custom_input_params):

    logging.warning(custom_input_params)

    mgs = custom_input_params["MODIF_GENOME_SIZE"]
    path = test_experiment_path / f"mgs={mgs}.yml"
    with open(path, "w") as file_:
        yaml.dump(custom_input_params, file_)

    try:
        run(
            custom_config_path=path,
            pickle_path=None,
            overwrite=True,
            custom_input_params=custom_input_params,
        )
    except Exception as e:
        raise AssertionError(f"run raised an exception for custom_input_params={custom_input_params}: {e}")


@pytest.mark.parametrize(
    "custom_input_params",
    [
        {
            # Basic
            "STEPS_PER_SIMULATION": 100,
            "LOGGING_RATE": 100,
            "SNAPSHOT_FINAL_COUNT": 3,
            # Tested
            "GENARCH_TYPE": "composite",
            "BITS_PER_LOCUS": bpl,
        }
        for bpl in [1, 2, 3, 4, 8, 20]
    ],
)
def test_BITS_PER_LOCUS(custom_input_params):

    logging.warning(custom_input_params)

    bpl = custom_input_params["BITS_PER_LOCUS"]
    path = test_experiment_path / f"bpl={bpl}.yml"
    with open(path, "w") as file_:
        yaml.dump(custom_input_params, file_)

    try:
        run(
            custom_config_path=path,
            pickle_path=None,
            overwrite=True,
            custom_input_params=custom_input_params,
        )
    except Exception as e:
        raise AssertionError(f"run raised an exception for custom_input_params={custom_input_params}: {e}")
