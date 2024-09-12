from aegis_gui.config.config import config
from aegis_gui.utilities import log_funcs, utilities


def is_input_in_valid_range(input_, param_name: str) -> bool:
    param = utilities.DEFAULT_PARAMETERS[param_name]
    input_ = param.convert(input_)
    in_inrange = param.inrange(input_)

    if config.ENVIRONMENT == "server":
        in_serverrange = param.serverrange(input_)
    else:
        in_serverrange = True

    return in_serverrange and in_inrange
