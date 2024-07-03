from typing import Union, Type, Optional


class BaseConfig:
    env: str
    debug_mode: bool
    loglevel: str
    simulation_number_limit: Optional[int]
    can_delete_default_sim: bool
    default_selection_states: tuple


class LocalConfig(BaseConfig):
    env = "local"
    debug_mode = True
    loglevel = "debug"
    simulation_number_limit = None
    can_delete_default_sim = False
    default_selection_states = (["default", True],)


class ServerConfig(BaseConfig):
    env = "server"
    debug_mode = False
    loglevel = "info"
    simulation_number_limit = 3
    can_delete_default_sim = False
    default_selection_states = (["default", True],)


config: Union[Type[LocalConfig], Type[ServerConfig], None] = None


def set(environment: str):
    global config
    if environment == "local":
        config = LocalConfig
    elif environment == "server":
        config = ServerConfig
    else:
        raise ValueError("Invalid environment")
