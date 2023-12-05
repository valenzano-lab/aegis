"""Universal configuration

Contains simulation-wide parameters (parameters that affect the whole simulation and not just one ecosystem).
Wraps also some other useful helper functions.
"""

import pathlib
import time
import numpy as np
import shutil
import logging
import yaml

from aegis.help import config
from aegis.help import other

from aegis.modules import popgenstats
from aegis.modules import recorder

from aegis.modules.genetics import phenomap
from aegis.modules.genetics import interpreter
from aegis.modules.genetics import flipmap


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s -- %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
    level=logging.DEBUG,
)

stage = 1
time_start = time.time()
here = pathlib.Path(__file__).absolute().parent


def init(custom_config_path, overwrite, running_on_server=False):
    global output_path
    global progress_path
    global params
    global LOGGING_RATE_
    global STAGES_PER_SIMULATION_

    # Output directory
    output_path = custom_config_path.parent / custom_config_path.stem
    if output_path.exists() and output_path.is_dir():
        if overwrite:
            shutil.rmtree(output_path)  # Delete previous directory if existing
        else:
            raise Exception(f"--overwrite is set to False but {output_path} already exists")
    output_path.mkdir(parents=True, exist_ok=True)

    # Get parameters
    params = get_params(custom_config_path, running_on_server=running_on_server)

    # Simulation-wide parameters
    STAGES_PER_SIMULATION_ = params["STAGES_PER_SIMULATION_"]
    LOGGING_RATE_ = params["LOGGING_RATE_"]
    recorder.PICKLE_RATE_ = params["PICKLE_RATE_"]
    recorder.SNAPSHOT_RATE_ = params["SNAPSHOT_RATE_"]
    recorder.VISOR_RATE_ = params["VISOR_RATE_"]
    recorder.POPGENSTATS_RATE_ = params["POPGENSTATS_RATE_"]
    popgenstats.POPGENSTATS_SAMPLE_SIZE_ = params["POPGENSTATS_SAMPLE_SIZE_"]

    # Init phenomap (partial)
    phenomap.PHENOMAP_SPECS = params["PHENOMAP_SPECS"]
    phenomap.PHENOMAP_METHOD = params["PHENOMAP_METHOD"]

    # Init interpreter
    interpreter.init(
        interpreter,
        BITS_PER_LOCUS=params["BITS_PER_LOCUS"],
        DOMINANCE_FACTOR=params["DOMINANCE_FACTOR"],
        THRESHOLD=params["THRESHOLD"],
    )

    # Init flipmap (partial)
    flipmap.FLIPMAP_CHANGE_RATE = params["FLIPMAP_CHANGE_RATE"]

    # Random number generator
    recorder.random_seed = np.random.randint(1, 10**6) if params["RANDOM_SEED_"] is None else params["RANDOM_SEED_"]
    other.rng = np.random.default_rng(recorder.random_seed)

    # Progress log
    progress_path = output_path / "progress.log"
    content = ("stage", "ETA", "t1M", "runtime", "stg/min")
    with open(progress_path, "wb") as f:
        np.savetxt(f, [content], fmt="%-10s", delimiter="| ")


def skip(rate):
    """Should you skip an action performed at a certain rate"""
    return (rate <= 0) or (stage % rate > 0)


def _log_progress():
    """Record some information about the time and speed of simulation."""

    if skip(LOGGING_RATE_):
        return

    logging.info("%8s / %s", stage, STAGES_PER_SIMULATION_)

    # Get time estimations
    time_diff = time.time() - time_start

    seconds_per_100 = time_diff / stage * 100
    eta = (STAGES_PER_SIMULATION_ - stage) / 100 * seconds_per_100

    stages_per_min = int(stage / (time_diff / 60))

    runtime = get_dhm(time_diff)
    time_per_1M = get_dhm(time_diff / stage * 1000000)
    eta = get_dhm(eta)

    # Save time estimations
    content = (stage, eta, time_per_1M, runtime, stages_per_min)
    with open(progress_path, "ab") as f:
        np.savetxt(f, [content], fmt="%-10s", delimiter="| ")


# Static methods


def get_dhm(timediff):
    """Format time in a human-readable format."""
    d = int(timediff / 86400)
    timediff %= 86400
    h = int(timediff / 3600)
    timediff %= 3600
    m = int(timediff / 60)
    return f"{d}`{h:02}:{m:02}"


def get_params(custom_config_path, running_on_server):
    """Fetch and validate input parameters."""
    # Read config parameters from the custom config file
    with open(custom_config_path, "r") as f:
        custom_config_params = yaml.safe_load(f)

    if custom_config_params is None:  # If config file is empty
        custom_config_params = {}

    # Read config parameters from the default config file
    default_config_params = config.get_default_parameters()

    # Fuse
    params = {}
    params.update(default_config_params)
    params.update(custom_config_params)

    config.validate(params, validate_resrange=running_on_server)

    return params
