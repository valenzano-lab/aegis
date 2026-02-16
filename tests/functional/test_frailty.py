import pytest
import pathlib
import logging
import yaml

from tests.functional.conftest import test_experiment_path

from aegis_sim import run  # Adjust the import to match your actual function location
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "custom_input_params",
    [
        {
            # Background
            "STEPS_PER_SIMULATION": 100,
            "INTERVAL_RATE": 100,
            "SNAPSHOT_FINAL_COUNT": 0,
            # Tested
            "FRAILTY_MODIFIER": f,
        }
        for f in DEFAULT_PARAMETERS["FRAILTY_MODIFIER"].evalrange
    ],
)
def test_frailty(custom_input_params):

    logging.warning(custom_input_params)

    f = custom_input_params["FRAILTY_MODIFIER"]
    path = test_experiment_path / f"f={f}.yml"
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
