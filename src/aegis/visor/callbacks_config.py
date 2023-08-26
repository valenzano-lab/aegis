from dash import html, dcc, callback, Output, Input, State, ALL, MATCH
from aegis.visor import funcs

from aegis.visor.callbacks_results import SELECTION


@callback(
    Output("config-make-text", "value"),
    Input("simulation-run-button", "n_clicks"),
    State("config-make-text", "value"),
    [
        State(f"config-{k}", "value")
        for k, v in funcs.DEFAULT_CONFIG_DICT.items()
        if not isinstance(v, list)
    ],
    prevent_initial_call=True,
)
@funcs.print_function_name
def run_simulation(n_clicks, filename, *values):
    if n_clicks is None:
        return

    # make config file
    custom_config = {
        k: val
        for (k, v), val in zip(funcs.DEFAULT_CONFIG_DICT.items(), values)
        if not isinstance(v, list)
    }
    funcs.make_config_file(filename, custom_config)

    SELECTION.add(filename)

    # run simulation
    funcs.run(filename)

    return ""


@callback(
    Output("simulation-run-button", "disabled"),
    Input("config-make-text", "value"),
)
@funcs.print_function_name
def block_sim_button(filename):

    if filename is None or filename == "":
        return True

    sim_exists = funcs.sim_exists(filename)
    if sim_exists:
        return True
    return False
