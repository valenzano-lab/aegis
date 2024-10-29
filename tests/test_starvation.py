import unittest
import pathlib
from aegis_sim import run
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
import yaml


def create_starvation_test(starvation_response):
    """Creates a unique test function for a specific starvation response value."""

    def test_func(self):
        # Construct the path for the YAML config file
        path = pathlib.Path(__file__).absolute().parent / f"STARVATION_RESPONSE={starvation_response}.yml"
        with open(path, "w") as file_:
            # Prepare data for YAML configuration
            data = {"STARVATION_RESPONSE": starvation_response, "STEPS_PER_SIMULATION": 100}
            if starvation_response == "cliff":
                data["CLIFF_SURVIVORSHIP"] = 0.5

            # Write data to YAML file
            yaml.dump(data=data, stream=file_)

        try:
            # Attempt to run the simulation with the created config
            run(
                custom_config_path=path,
                pickle_path=None,
                overwrite=True,
                custom_input_params={},
            )
        except Exception as e:
            # Raise an AssertionError if the run fails
            raise AssertionError(
                f"aegis_sim.run raised an exception for STARVATION_RESPONSE={starvation_response}: {e}"
            )

    return test_func


class TestAegisSim(unittest.TestCase):
    pass


# Dynamically create a test case for each STARVATION_RESPONSE value
for starvation_response in DEFAULT_PARAMETERS["STARVATION_RESPONSE"].evalrange:
    test_name = f"test_starvation_response_{starvation_response.replace(' ', '_')}"  # Replace spaces with underscores for valid function names
    test_func = create_starvation_test(starvation_response)
    setattr(TestAegisSim, test_name, test_func)


if __name__ == "__main__":
    unittest.main()
