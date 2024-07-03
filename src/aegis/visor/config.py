config = None


class LocalConfig:
    env = "local"
    debug = True
    loglevel = "debug"
    simulation_number_limit = None


class ServerConfig:
    env = "server"
    debug = False
    loglevel = "info"
    simulation_number_limit = 3


def set(environment):
    global config
    if environment == "local":
        config = LocalConfig
    elif environment == "server":
        config = ServerConfig
    else:
        raise Exception
