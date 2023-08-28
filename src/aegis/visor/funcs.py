import platformdirs
import subprocess

import time
import pathlib
import yaml
import logging

from aegis.parameters import param

logging.basicConfig(level=logging.INFO)

BASE_DIR = pathlib.Path(platformdirs.user_data_dir("aegis", "aegis"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

HERE = pathlib.Path(__file__).absolute().parent


def read_yml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


DEFAULT_CONFIG_DICT = read_yml(HERE.parent / "parameters/default.yml")


def get_config_path(filename):
    return BASE_DIR / f"{filename}.yml"


def run(filename):
    subprocess.run(
        ["python3", "-m", "aegis", "--config_path", get_config_path(filename)],
        check=True,
    )


def make_config_file(filename, configs):
    configs["PHENOMAP_SPECS"] = []
    configs["NOTES"] = []
    for k, v in configs.items():
        configs[k] = param.params[k].convert(v)
    logging.info("making config file")
    config_path = get_config_path(filename)
    with open(config_path, "w") as file_:
        yaml.dump(configs, file_)


def get_default_config_dict():
    return read_yml(HERE.parent / "parameters/default.yml")


def print_function_name(func):
    def wrapper(*args, **kwargs):
        print(f"Executing function: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def get_sim_paths():
    return [p for p in BASE_DIR.iterdir() if p.is_dir()]


def sim_exists(filename: str) -> bool:
    paths = get_sim_paths()
    return any(path.stem == filename for path in paths)
