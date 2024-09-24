import yaml
import pathlib
from typing import Optional


class GuiSettings:
    ENVIRONMENT: str
    DEBUG_MODE: bool
    # loglevel: str
    SIMULATION_NUMBER_LIMIT: Optional[int]
    # can_delete_default_sim: bool
    # default_selection_states: tuple
    DATA_RETENTION_DAYS: Optional[int]
    ABS_PATH_TO_BASE_DIRECTORY: Optional[str]
    PORT: Optional[int]
    BASE_HREF = "/aegis/"

    def can_run_more_simulations(self, currently_running):
        if self.SIMULATION_NUMBER_LIMIT is None:
            return True
        else:
            return self.SIMULATION_NUMBER_LIMIT > currently_running

    def get_base_dir(self):
        if self.ABS_PATH_TO_BASE_DIRECTORY is None:
            return pathlib.Path.home().absolute()
        else:
            return pathlib.Path(self.ABS_PATH_TO_BASE_DIRECTORY).absolute()

    def set(self, environment: str, debug):

        # Read from YML
        path_to_yaml = pathlib.Path(__file__).parent / "gui_settings.yml"
        with open(path_to_yaml, "r") as file:
            yml = yaml.safe_load(file)
        if environment == "local":
            yml = yml["LocalConfig"]
        elif environment == "server":
            yml = yml["ServerConfig"]
        else:
            raise ValueError(f"{environment} is an invalid input for environment; should be 'local' or 'server'")

        assert isinstance(debug, bool)

        self.ENVIRONMENT = yml["ENVIRONMENT"]
        self.DEBUG_MODE = debug
        self.SIMULATION_NUMBER_LIMIT = yml["SIMULATION_NUMBER_LIMIT"]
        self.DATA_RETENTION_DAYS = yml["DATA_RETENTION_DAYS"]
        self.MAX_SIM_NAME_LENGTH = yml["MAX_SIM_NAME_LENGTH"]
        self.ABS_PATH_TO_BASE_DIRECTORY = yml["ABS_PATH_TO_BASE_DIRECTORY"]
        self.PORT = yml["PORT"]

        # Derived
        self.base_dir = self.get_base_dir() / "aegis_data"
        self.sim_dir = self.base_dir / "sim_data"
        self.figure_dir = self.base_dir / "figures"

        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.sim_dir.mkdir(exist_ok=True, parents=True)
        self.figure_dir.mkdir(exist_ok=True, parents=True)

    def wrap_href(self, href):
        return self.BASE_HREF + href


gui_settings = GuiSettings()

# def create_config_class(name, attrs):
#     """Dynamically create a configuration class."""
#     return type(name, (BaseConfig,), attrs)


# def load_configs_from_yaml(file_path):

#     config_classes = {}

#     for config_name, config_attrs in configs.items():
#         config_class = create_config_class(config_name, config_attrs)
#         config_classes[config_name] = config_class

#     return config_classes


# LocalConfig = config_classes["LocalConfig"]
# ServerConfig = config_classes["ServerConfig"]
