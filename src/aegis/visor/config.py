config = None


class LocalConfig:
    env = "local"
    debug = True
    loglevel = "debug"
    simulation_number_limit = None
    can_delete_default_sim = False
    default_selection_states = (["default", True],)


class ServerConfig:
    env = "server"
    debug = False
    loglevel = "info"
    simulation_number_limit = 3
    can_delete_default_sim = False
    default_selection_states = (["default", True],)


def set(environment):
    global config
    if environment == "local":
        config = LocalConfig
    elif environment == "server":
        config = ServerConfig
    else:
        raise Exception
