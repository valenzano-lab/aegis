import unittest
import pathlib
import shutil
import yaml
from aegis_sim import run
from tests.functional.conftest import test_experiment_path


class TestAegisSim(unittest.TestCase):

    def test_run_simulation(self):
        # Write config into experiments dir so output lands there too
        config = {
            "STEPS_PER_SIMULATION": 100,
            "LOGGING_RATE": 100,
            "SNAPSHOT_FINAL_COUNT": 3,
        }
        path = test_experiment_path / "test_sim.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config, f)

        try:
            run(custom_config_path=path, pickle_path=None, overwrite=True, custom_input_params={})
        except Exception as e:
            self.fail(f"aegis_sim.run raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
