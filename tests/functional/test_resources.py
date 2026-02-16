import pytest
import pathlib
import logging
import yaml

from aegis_sim import run  # Adjust the import to match your actual function location
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_sim.utilities.container import Container
from tests.functional.conftest import test_experiment_path

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "custom_input_params",
    [
        {
            # Background
            "STEPS_PER_SIMULATION": 100,
            "SNAPSHOT_FINAL_COUNT": 0,
            # Tested
            "STARVATION_MORTALITY_FACTOR": sce,
            # "STARVATION_MORTALITY_MAXIMUM": smm,
            # "STARVATION_PREVENT_OVERREACTION": spo,
        }
        for sce in DEFAULT_PARAMETERS["STARVATION_MORTALITY_FACTOR"].evalrange
        # for smm in DEFAULT_PARAMETERS["STARVATION_MORTALITY_MAXIMUM"].evalrange
        # for spo in DEFAULT_PARAMETERS["STARVATION_PREVENT_OVERREACTION"].evalrange
    ],
)
def test_STARVATION(custom_input_params):

    logging.warning(custom_input_params)

    sce = custom_input_params["STARVATION_MORTALITY_FACTOR"]
    # smm = custom_input_params["STARVATION_MORTALITY_MAXIMUM"]
    # spo = custom_input_params["STARVATION_PREVENT_OVERREACTION"]
    path = test_experiment_path / f"sce={str(sce)}.yml"
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
