from aegis.visor.config import config
from aegis.visor.utilities import log_funcs, utilities


def is_input_in_valid_range(input_, param_name: str) -> bool:
    param = utilities.DEFAULT_PARAMETERS[param_name]
    input_ = param.convert(input_)
    in_inrange = param.inrange(input_)

    if config.env == "server":
        in_serverrange = param.serverrange(input_)
    else:
        in_serverrange = True

    return in_serverrange and in_inrange
