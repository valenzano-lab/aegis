import yaml
import logging

from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_gui.utilities.utilities import get_config_path, get_container, safe_sim_name, safe_sim_path
from aegis_sim.utilities.container import Container
import subprocess

# NOTE: this dict is per-process. On a multi-worker server (gunicorn etc.) only
# the worker that started a sim will see it here. Termination uses Container.terminate()
# via ticker files instead, which works across workers.
running_processes = {}


def make_config_file(filename, configs):
    safe_sim_name(filename)
    configs["PHENOMAP_SPECS"] = []
    configs["NOTES"] = []
    for k, v in configs.items():
        configs[k] = DEFAULT_PARAMETERS[k].convert(v)
    logging.info("Making a config file.")
    config_path = get_config_path(filename)
    with open(config_path, "w") as file_:
        yaml.dump(configs, file_)


def run_simulation(filename, prerun_sim_path):
    global running_processes
    safe_sim_name(filename)
    config_path = get_config_path(filename)
    logging.info(f"Running a simulation at path {config_path}.")
    if prerun_sim_path is None:
        pickle_command = []
    else:
        # Validate the prerun sim is one of ours — never trust a raw path string.
        prerun_path = safe_sim_path(_extract_sim_name(prerun_sim_path))
        container = Container(prerun_path)
        latest_pickle_path = container.get_path(["pickles"])[-1]
        logging.info(f"Using pickled population from {latest_pickle_path}.")
        pickle_command = ["-p", str(latest_pickle_path)]
    process = subprocess.Popen(
        ["aegis", "sim", "--config_path", str(config_path)] + pickle_command
    )  # used to use sys.executable for cross-platform 'python3' command
    running_processes[filename] = process


def _extract_sim_name(value):
    """Pull the bare sim name out of whatever the caller handed us — could be a
    name, a relative path, or an absolute path. We only ever use the stem."""
    import pathlib

    return pathlib.Path(str(value)).name


def terminate_simulation(simname):
    safe_sim_name(simname)
    container = get_container(filename=simname)
    container.terminate()
