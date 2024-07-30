from dash import html, dcc, callback, Output, Input, State, ALL
from aegis_gui.utilities import log_funcs, utilities, ps_list
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS

from aegis_gui.config import config

from .valid_range import is_input_in_valid_range

import logging

layout = html.Div(
    id="sim-section-control",
    children=[
        dcc.Input(
            id="config-make-text",
            className="control-element",
            type="text",
            placeholder="unique id",
            autoComplete="off",
        ),
        html.Button(
            "run simulation",
            id="simulation-run-button",
            className="control-element",
        ),
        html.P("", id="simulation-run-text"),
        # html.Button("make config", id="config-make-button"),]
    ],
)


def decode_config_tab_values(values, ids_):
    for value, id_ in zip(values, ids_):
        param = DEFAULT_PARAMETERS[id_["index"]]
        if param.dtype == bool:
            yield id_["index"], bool(value)
        else:
            yield id_["index"], value


@callback(
    Output("config-make-text", "value"),
    Input("simulation-run-button", "n_clicks"),
    State("config-make-text", "value"),
    State({"type": "config-input", "index": ALL}, "value"),
    State({"type": "config-input", "index": ALL}, "id"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def click_sim_button(n_clicks, filename, values, ids_):
    """
    Run simulation when sim button clicked (also, implicitly, not disabled).
    """
    if n_clicks is None:
        return

    # make config file

    decoded_pairs = list(decode_config_tab_values(values=values, ids_=ids_))
    input_config = {i: v for i, v in decoded_pairs}
    utilities.make_config_file(filename, input_config)

    # run simulation
    utilities.run_simulation(filename)
    return ""


def is_sim_name_valid(sim_name: str) -> bool:
    return (sim_name is not None) and (sim_name != "") and ("." not in sim_name)


@callback(
    Output("simulation-run-button", "disabled"),
    Input("config-make-text", "value"),
    Input({"type": "config-input", "index": ALL}, "value"),
    State({"type": "config-input", "index": ALL}, "id"),
)
@log_funcs.log_debug
def disable_sim_button(filename, values, ids) -> bool:
    """
    Make simulation run button unclickable when any of these is true:
    - input values for parameters are not inside the acceptable server range
    - simulation name is not valid
    - simulation name is already used
    """

    for value, id_ in zip(values, ids):
        param_name = id_["index"]
        is_value_set = value != "" and value is not None
        if not is_value_set:
            continue

        # check validity of input value
        in_valid_range = is_input_in_valid_range(input_=value, param_name=param_name)
        if not in_valid_range:
            logging.info(
                f"Simulation run button is blocked because parameter {param_name} has received an invalid input."
            )
            return True

    if not is_sim_name_valid(filename):
        logging.info(f"Simulation run button is blocked because simulation name {filename} is not valid.")
        return True

    if utilities.sim_exists(filename):
        logging.info(f"Simulation run button is blocked because simulation name {filename} already exists.")
        return True

    # Check if reached simulation number limit
    currently_running = len(ps_list.run_ps_af())
    if not config.can_run_more_simulations(currently_running=currently_running):
        logging.info(
            f"You are currently running {currently_running} simulations. Limit is {config.simulation_number_limit}."
        )
        return True

    return False
