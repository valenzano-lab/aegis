import pathlib
import time
import numpy as np
import shutil
import argparse
import logging
import yaml


from aegis.parameters import validate


logging.basicConfig(
    format="%(asctime)s : %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
    level=logging.INFO,
)


def get_dhm(timediff):
    """Format time in a human-readable format."""
    d = int(timediff / 86400)
    timediff %= 86400
    h = int(timediff / 3600)
    timediff %= 3600
    m = int(timediff / 60)
    return f"{d}`{h:02}:{m:02}"


class Panconfiguration:
    """Universal configuration

    Contains simulation-wide parameters (parameters that affect the whole simulation and not just one ecosystem).
    Wraps also some other useful helper functions.
    """
    def skip(self, rate):
        """Should you skip an action performed at a certain rate"""
        return (rate <= 0) or (self.stage % rate > 0)

    def __init__(self):
        pass

    def init(self, custom_config_path=None):
        self.here = pathlib.Path(__file__).absolute().parent
        self.stage = 0
        self.time_start = time.time()

        def read_yml(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)

        def run_parser():

            parser = argparse.ArgumentParser(
                description="Ageing of Evolving Genomes In Silico"
            )
            parser.add_argument(
                "custom_config_path",
                type=str,
                help="path to config file",
            )

            return parser.parse_args()

        def get_params(custom_config_path):
            """Fetch and validate input parameters."""
            # Read config parameters from the custom config file
            custom_config_params = read_yml(custom_config_path)

            if custom_config_params is None:  # If config file is empty
                custom_config_params = {}

            # Read config parameters from the default config file
            default_config_params = read_yml(self.here / "parameters/default.yml")

            # Fuse
            params = {}
            params.update(default_config_params)
            params.update(custom_config_params)

            validate.validate_keys(params, list(default_config_params.keys()))
            validate.validate_values(params)

            return params

        # Get path to custom configuration file
        if custom_config_path is None:
            parsed_args = run_parser()
            custom_config_path = pathlib.Path(parsed_args.custom_config_path).absolute()

        logging.info("Custom config path = %s", custom_config_path)

        # Get parameters
        params = get_params(custom_config_path)
        self.params_list = (params,)

        # Simulation-wide parameters
        self.ECOSYSTEM_NUMBER_ = params["ECOSYSTEM_NUMBER_"]
        self.STAGES_PER_SIMULATION_ = params["STAGES_PER_SIMULATION_"]
        self.LOGGING_RATE_ = params["LOGGING_RATE_"]
        # self.PICKLE_RATE_ = params["PICKLE_RATE_"]
        self.SNAPSHOT_RATE_ = params["SNAPSHOT_RATE_"]
        self.VISOR_RATE_ = params["VISOR_RATE_"]
        self.POPGENSTATS_RATE_ = params["POPGENSTATS_RATE_"]

        # Random number generator
        self.random_seed = (
            np.random.randint(1, 10 ** 6)
            if params["RANDOM_SEED_"] is None
            else params["RANDOM_SEED_"]
        )
        self.rng = np.random.default_rng(self.random_seed)

        # Output directory
        self.output_path = custom_config_path.parent / custom_config_path.stem
        if self.output_path.exists() and self.output_path.is_dir():
            shutil.rmtree(self.output_path)  # Delete previous directory if existing
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Progress log
        self.progress_path = self.output_path / "progress.log"
        content = ("stage", "ETA", "t1M", "runtime", "stg/min")
        with open(self.progress_path, "wb") as f:
            np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

    def _log_progress(self):
        """Record some information about the time and speed of simulation."""
        logging.info("%8s / %s", self.stage, self.STAGES_PER_SIMULATION_)

        # Get time estimations
        time_diff = time.time() - self.time_start

        seconds_per_100 = time_diff / self.stage * 100
        eta = (self.STAGES_PER_SIMULATION_ - self.stage) / 100 * seconds_per_100

        stages_per_min = int(self.stage / (time_diff / 60))

        runtime = get_dhm(time_diff)
        time_per_1M = get_dhm(time_diff / self.stage * 1000000)
        eta = get_dhm(eta)

        # Save time estimations
        content = (self.stage, eta, time_per_1M, runtime, stages_per_min)
        with open(self.progress_path, "ab") as f:
            np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

    def run_stage(self):
        """
        1) Increment stage
        2) Log progress
        3) Return whether the simulation should continue
        """

        # Increment stage
        self.stage += 1

        # Log progress

        if not self.skip(self.LOGGING_RATE_):
            self._log_progress()

        # Return True if the simulation is to be continued
        return self.stage < self.STAGES_PER_SIMULATION_


pan = Panconfiguration()  # Initialize now, set up later
