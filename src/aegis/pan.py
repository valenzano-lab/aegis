import pathlib
import time
import numpy as np
import shutil
import logging
import argparse
import yaml

from aegis import cnf, var
from aegis.help.config import get_default_parameters, validate

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s -- %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
    level=logging.DEBUG,
)


def parse():
    # Define parser
    parser = argparse.ArgumentParser(description="Aging of Evolving Genomes in Silico")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path to config file",
        default="",
    )
    parser.add_argument(
        "-p",
        "--pickle_path",
        type=str,
        help="path to pickle file",
        default="",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        help="overwrite old data with new simulation",
        default=False,
    )

    # Parse arguments
    args = parser.parse_args()

    # Decision tree
    config_path = pathlib.Path(args.config_path).absolute() if args.config_path else ""
    pickle_path = pathlib.Path(args.pickle_path).absolute() if args.pickle_path else ""

    return config_path, pickle_path, args.overwrite


def set_up_cnf(custom_config_path, running_on_server):
    # Read config parameters from the custom config file
    if custom_config_path == "":
        custom_config_params = {}
    else:
        with open(custom_config_path, "r") as f:
            custom_config_params = yaml.safe_load(f)
        if custom_config_params is None:  # If config file is empty
            custom_config_params = {}

    # Read config parameters from the default config file
    default_config_params = get_default_parameters()

    # Fuse
    params = {}
    params.update(default_config_params)
    params.update(custom_config_params)

    validate(params, validate_resrange=running_on_server)

    for k, v in params.items():
        setattr(cnf, k, v)


def skip(rate):
    """Should you skip an action performed at a certain rate"""
    return (rate <= 0) or (var.stage % rate > 0)


# Decorators
def profile_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print(f"{func.__name__} took {execution_time:.6f} ms to execute")
        return result

    return wrapper


# INITIALIZE

time_start = time.time()
here = pathlib.Path(__file__).absolute().parent
config_path, pickle_path, overwrite = parse()


if config_path:
    set_up_cnf(config_path, running_on_server=False)
    # Output directory
    output_path = config_path.parent / config_path.stem
    if output_path.exists() and output_path.is_dir():
        if overwrite:
            shutil.rmtree(output_path)  # Delete previous directory if existing
        else:
            raise Exception(f"--overwrite is set to False but {output_path} already exists")
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up progress log
    progress_path = output_path / "progress.log"
    content = ("stage", "ETA", "t1M", "runtime", "stg/min", "popsize")
    with open(progress_path, "wb") as f:
        np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

    # Set up random number generator
    random_seed = np.random.randint(1, 10**6) if cnf.RANDOM_SEED_ is None else cnf.RANDOM_SEED_
    var.rng = np.random.default_rng(random_seed)

    # Set up season
    season_countdown = float("inf") if cnf.STAGES_PER_SEASON == 0 else cnf.STAGES_PER_SEASON
