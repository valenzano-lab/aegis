from dash import callback, Output, Input, State, ALL, MATCH, ctx
from visor import utilities
import logging
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS


@callback(
    Output("config-make-text", "value"),
    Input("simulation-run-button", "n_clicks"),
    State("config-make-text", "value"),
    State({"type": "config-input", "index": ALL}, "value"),
    prevent_initial_call=True,
)
@utilities.log_debug
def click_sim_button(n_clicks, filename, values):
    """
    Run simulation when sim button clicked (also, implicitly, not disabled).
    """
    if n_clicks is None:
        return

    # make config file
    default_config = DEFAULT_PARAMETERS.copy()
    custom_config = {k: val for (k, v), val in zip(default_config.items(), values) if not isinstance(v, list)}
    utilities.make_config_file(filename, custom_config)

    # run simulation
    utilities.run(filename)

    return ""


def is_sim_name_valid(sim_name: str) -> bool:
    return (sim_name is not None) and (sim_name != "") and ("." not in sim_name)


def is_input_in_serverrange(input_, param_name: str) -> bool:
    param = utilities.DEFAULT_PARAMETERS[param_name]
    input_ = param.convert(input_)
    in_serverrange = param.serverrange(input_)
    return in_serverrange


@callback(
    Output("simulation-run-button", "disabled"),
    Input("config-make-text", "value"),
    Input({"type": "config-input", "index": ALL}, "value"),
)
@utilities.log_debug
def disable_sim_button(filename, values) -> bool:
    """
    Make simulation run button unclickable when any of these is true:
    - input values for parameters are not inside the acceptable server range
    - simulation name is not valid
    - simulation name is already used
    """
    default_config = DEFAULT_PARAMETERS.copy()
    for k, value in zip(default_config, values):

        is_value_set = value != "" and value is not None
        if not is_value_set:
            continue

        # check validity of input value
        in_serverrange = is_input_in_serverrange(input_=value, param_name=k)
        if not in_serverrange:
            logging.info(f"Simulation run button is blocked because parameter {k} has received an invalid input")
            return True

    if not is_sim_name_valid(filename):
        logging.info(f"Simulation run button is blocked because simulation name {filename} is not valid")
        return True

    if utilities.sim_exists(filename):
        logging.info(f"Simulation run button is blocked because simulation name {filename} already exists")
        return True

    return False


@callback(
    Output({"type": "config-input", "index": MATCH}, "className"),
    Input({"type": "config-input", "index": MATCH}, "value"),
    State({"type": "config-input", "index": MATCH}, "className"),
    prevent_initial_call=True,
)
@utilities.log_debug
def disable_config_input(value, className) -> str:
    """
    Change style of config input so that the user knows that the input value is outside of valid server range.
    """
    if ctx.triggered_id is None:
        return className

    param_name = ctx.triggered_id["index"]
    className = className.replace(" disabled", "")

    print(param_name)
    in_serverrange = is_input_in_serverrange(input_=value, param_name=param_name)
    if not in_serverrange:
        className += " disabled"

    return className
