from typing import Union, Type, Optional


class BaseConfig:
    env: str
    debug_mode: bool
    loglevel: str
    simulation_number_limit: Optional[int]
    can_delete_default_sim: bool
    default_selection_states: tuple
    data_retention_days: int

    def can_run_more_simulations(self, currently_running):
        if self.simulation_number_limit is None:
            return True
        else:
            return self.simulation_number_limit > currently_running


class LocalConfig(BaseConfig):
    env = "local"
    debug_mode = True
    loglevel = "debug"
    simulation_number_limit = None
    can_delete_default_sim = False
    default_selection_states = (["default", True],)
    data_retention_days = 365


class ServerConfig(BaseConfig):
    env = "server"
    debug_mode = False
    loglevel = "info"
    simulation_number_limit = 3
    can_delete_default_sim = False
    default_selection_states = (["default", True],)
    data_retention_days = (
        7  # input_summary.json, created during simulation initialization, contains the time when a simulation started
    )


config: Union[Type[LocalConfig], Type[ServerConfig], None] = None


def set(environment: str):
    global config
    if environment == "local":
        config = LocalConfig()
    elif environment == "server":
        config = ServerConfig()
    else:
        raise ValueError("Invalid environment")
