class DevConfig:
    debug = True
    loglevel = "debug"


class ServerConfig:
    debug = False
    loglevel = "info"


def get_config(environment):
    if environment == "dev":
        return DevConfig
    elif environment == "server":
        return ServerConfig
    else:
        raise Exception
