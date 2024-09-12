import yaml
import pathlib
from typing import Optional


class BaseConfig:
    ENVIRONMENT: str
    DEBUG_MODE: bool
    # loglevel: str
    SIMULATION_NUMBER_LIMIT: Optional[int]
    # can_delete_default_sim: bool
    # default_selection_states: tuple
    DATA_RETENTION_DAYS: int

    def can_run_more_simulations(self, currently_running):
        if self.SIMULATION_NUMBER_LIMIT is None:
            return True
        else:
            return self.SIMULATION_NUMBER_LIMIT > currently_running


def create_config_class(name, attrs):
    """Dynamically create a configuration class."""
    return type(name, (BaseConfig,), attrs)


def load_configs_from_yaml(file_path):
    with open(file_path, "r") as file:
        configs = yaml.safe_load(file)

    config_classes = {}

    for config_name, config_attrs in configs.items():
        config_class = create_config_class(config_name, config_attrs)
        config_classes[config_name] = config_class

    return config_classes


path_to_yaml = pathlib.Path(__file__).parent / "config.yml"
config_classes = load_configs_from_yaml(path_to_yaml)

LocalConfig = config_classes["LocalConfig"]
ServerConfig = config_classes["ServerConfig"]


def set(environment: str):
    global config
    if environment == "local":
        config = LocalConfig()
    elif environment == "server":
        config = ServerConfig()
    else:
        raise ValueError(f"{environment} is an invalid input for environment; should be 'local' or 'server'")
