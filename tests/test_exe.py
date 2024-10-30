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
    path = pathlib.Path(__file__).absolute().parent / "_.yml"
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
        {"MATURATION_AGE": v, "STEPS_PER_SIMULATION": 100, "SNAPSHOT_FINAL_COUNT": 0}
        for v in range(*DEFAULT_PARAMETERS["MATURATION_AGE"].evalrange, 10)
    ],
)
def test_MATURATION_AGE(custom_input_params):
    path = pathlib.Path(__file__).absolute().parent / "_.yml"
    logging.info(custom_input_params)
    with open(path, "w") as file_:
        file_.write("")
    try:
        run(
            custom_config_path=path,
            pickle_path=None,
            overwrite=True,
            custom_input_params=custom_input_params,
        )
        container = Container(str(path).strip(".yml"))
        output_summary = container.get_output_summary()
        logging.warning(output_summary)
    except Exception as e:
        raise AssertionError(f"run raised an exception for custom_input_params={custom_input_params}: {e}")
