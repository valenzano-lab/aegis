import platformdirs

import time
import pathlib
import yaml
import logging

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


def run():
    t = time.time()
    # subprocess.run("python3 -m aegis temp/longg.yml", shell=True, check=True)
    # make_config_file()
    print(time.time() - t)


def make_config_file(filename, configs):
    configs["PHENOMAP_SPECS"] = []
    configs["NOTES"] = []
    logging.info("making config file")
    with open(BASE_DIR / f"{filename}.yml", "w") as file_:
        yaml.dump(configs, file_)


def get_default_config_dict():
    return read_yml(HERE.parent / "parameters/default.yml")
