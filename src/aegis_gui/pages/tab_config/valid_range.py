from aegis_gui.guisettings.GuiSettings import gui_settings
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS


def is_input_in_valid_range(input_, param_name: str) -> bool:
    param = DEFAULT_PARAMETERS[param_name]
    input_ = param.convert(input_)
    in_inrange = param.inrange(input_)

    if gui_settings.ENVIRONMENT == "server":
        in_serverrange = param.serverrange(input_)
    else:
        in_serverrange = True

    return in_serverrange and in_inrange
