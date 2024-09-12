import yaml
import logging

from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_gui.utilities.utilities import get_config_path, get_container
from aegis_sim.utilities.container import Container
import subprocess

running_processes = {}


def make_config_file(filename, configs):
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
    config_path = get_config_path(filename)
    logging.info(f"Running a simulation at path {config_path}.")
    if prerun_sim_path is None:
        pickle_command = []
    else:
        container = Container(prerun_sim_path)
        latest_pickle_path = container.get_path(["pickles"])[-1]
        logging.info(f"Using pickled population from {latest_pickle_path}.")
        pickle_command = ["-p", latest_pickle_path]
    process = subprocess.Popen(
        ["aegis", "sim", "--config_path", config_path] + pickle_command
    )  # used to use sys.executable for cross-platform 'python3' command
    running_processes[filename] = process


def terminate_simulation(simname):
    container = get_container(filename=simname)
    container.terminate()
