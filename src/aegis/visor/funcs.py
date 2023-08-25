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

# for k, v in DEFAULT_CONFIG_DICT.items():
# print(isinstance(v, list))
# print()


def hello():
    print(BASE_DIR)


def run(filename):
    t = time.time()
    config_path = BASE_DIR / f"{filename}.yml"
    subprocess.run(f"python3 -m aegis {config_path}", shell=True, check=True)
    print(time.time() - t)


def make_config_file(filename, configs):
    configs["PHENOMAP_SPECS"] = []
    configs["NOTES"] = []
    for k, v in configs.items():
        configs[k] = param.params[k].convert(v)
    logging.info("making config file")
    config_path = BASE_DIR / f"{filename}.yml"
    with open(config_path, "w") as file_:
        yaml.dump(configs, file_)


def get_default_config_dict():
    return read_yml(HERE.parent / "parameters/default.yml")
