import pytest
import pathlib
import logging

from aegis_sim import run  # Adjust the import to match your actual function location
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_sim.utilities.container import Container

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "custom_input_params",
    [
        {"STARVATION_RESPONSE": starvation_response, "STEPS_PER_SIMULATION": 100, "SNAPSHOT_FINAL_COUNT": 0}
        for starvation_response in DEFAULT_PARAMETERS["STARVATION_RESPONSE"].evalrange
    ],
)
def test_STARVATION_RESPONSE(custom_input_params):
    path = (
        pathlib.Path(__file__).absolute().parent
        / f"STARVATION_RESPONSE={custom_input_params['STARVATION_RESPONSE']}.yml"
    )
    logging.info(custom_input_params)
    if custom_input_params["STARVATION_RESPONSE"] == "cliff":
        custom_input_params["CLIFF_SURVIVORSHIP"] = 0.5
    with open(path, "w") as file_:
        file_.write("")
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
            "STARVATION_RESPONSE": starvation_response,
            "STEPS_PER_SIMULATION": 100,
            "RESOURCE_MULTIPLICATIVE_GROWTH": resource_multiplicative_growth,
        }
        for starvation_response in ["worsening_proportional", "gradual", "treadmill_random"]
        for resource_multiplicative_growth in DEFAULT_PARAMETERS["RESOURCE_MULTIPLICATIVE_GROWTH"].evalrange
    ],
)
def test_RESOURCE_MULTIPLICATIVE_GROWTH(custom_input_params):
    path = (
        pathlib.Path(__file__).absolute().parent
        / f"RESOURCE_MULTIPLICATIVE_GROWTH={custom_input_params['RESOURCE_MULTIPLICATIVE_GROWTH']};STARVATION_RESPONSE={custom_input_params['STARVATION_RESPONSE']}.yml"
    )
    logging.info(custom_input_params)
    if custom_input_params["STARVATION_RESPONSE"] == "cliff":
        custom_input_params["CLIFF_SURVIVORSHIP"] = 0.5
    with open(path, "w") as file_:
        file_.write("")
    try:
        run(
            custom_config_path=path,
            pickle_path=None,
            overwrite=True,
            custom_input_params=custom_input_params,
        )
    except Exception as e:
        raise AssertionError(f"run raised an exception for custom_input_params={custom_input_params}: {e}")
