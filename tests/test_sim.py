import unittest
import pathlib
from aegis_sim import run

test_experiment_path = pathlib.Path(__file__).absolute().parent / "experiments"
test_experiment_path.mkdir(exist_ok=True)


class TestAegisSim(unittest.TestCase):

    def test_run_simulation(self):

        path = test_experiment_path.parent / "test_sim.yml"

        try:
            run(custom_config_path=path, pickle_path=None, overwrite=True, custom_input_params={})
        except Exception as e:
            self.fail(f"aegis_sim.run raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
