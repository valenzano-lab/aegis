import platformdirs
import subprocess

import pathlib
import yaml
import logging

from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS

# TODO ensure that there is default dataset available
default_selection_states = (["default", True],)


def get_here():
    return pathlib.Path(__file__).absolute().parent


def get_base_dir():
    return pathlib.Path(platformdirs.user_data_dir("aegis", "aegis"))


def get_sim_dir():
    base_dir = get_base_dir() / "sim_data"
    base_dir.mkdir(exist_ok=True)
    return base_dir


def get_figure_dir():
    figure_dir = get_base_dir() / "figures"
    figure_dir.mkdir(exist_ok=True)
    return figure_dir


def read_yml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config_path(filename):
    return get_sim_dir() / f"{filename}.yml"


def run(filename):
    config_path = get_config_path(filename)
    logging.info(f"Running a simulation at path {config_path}")
    print(f"Running a simulation at path {config_path}")
    subprocess.Popen(["python3", "-m", "aegis", "--config_path", config_path])


def make_config_file(filename, configs):
    configs["PHENOMAP_SPECS"] = []
    configs["NOTES"] = []
    for k, v in configs.items():
        configs[k] = DEFAULT_PARAMETERS[k].convert(v)
    logging.info("making config file")
    config_path = get_config_path(filename)
    with open(config_path, "w") as file_:
        yaml.dump(configs, file_)


def log_debug(func):
    def wrapper(*args, **kwargs):
        logging.debug(f"executing function: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def log_info(func):
    def wrapper(*args, **kwargs):
        logging.info(f"executing function: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def get_sim_paths():
    sim_dir = get_sim_dir()
    return [p for p in sim_dir.iterdir() if p.is_dir()]


def get_sims():
    return [p.stem for p in get_sim_paths()]


def sim_exists(filename: str) -> bool:
    paths = get_sim_paths()
    return any(path.stem == filename for path in paths)


def extract_visor_from_docstring(class_):
    docstring = class_.__doc__
    visor_section = docstring.split("VISOR")[1]
    return visor_section
