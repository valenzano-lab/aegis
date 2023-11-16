from dash import html, dcc, callback, Output, Input, State, ALL, MATCH, ctx
from aegis.visor import funcs
from aegis.help import config

DEFAULT_CONFIG_DICT = config.get_default_parameters()


@callback(
    Output("config-make-text", "value"),
    Input("simulation-run-button", "n_clicks"),
    State("config-make-text", "value"),
    State({"type": "config-input", "index": ALL}, "value"),
    # [
    #     State(f"config-{k}", "value")
    #     for k, v in DEFAULT_CONFIG_DICT.items()
    #     if not isinstance(v, list)
    # ],
    prevent_initial_call=True,
)
@funcs.print_function_name
def run_simulation(n_clicks, filename, values):
    if n_clicks is None:
        return

    # make config file
    custom_config = {
        k: val
        for (k, v), val in zip(DEFAULT_CONFIG_DICT.items(), values)
        if not isinstance(v, list)
    }
    funcs.make_config_file(filename, custom_config)

    # run simulation
    funcs.run(filename)

    return ""


@callback(
    Output("simulation-run-button", "disabled"),
    Input("config-make-text", "value"),
    Input({"type": "config-input", "index": ALL}, "value"),
    # [
    #     Input(f"config-{k}", "value")
    #     for k, v in DEFAULT_CONFIG_DICT.items()
    #     if not isinstance(v, list)
    # ],
)
@funcs.print_function_name
def block_sim_button(filename, values):
    for k, value in zip(DEFAULT_CONFIG_DICT, values):
        if value != "" and value is not None:
            param = config.params[k]
            val = param.convert(value)
            valid = param.resrange(val)
            if not valid:
                return True

    if filename is None or filename == "" or "." in filename:
        return True

    sim_exists = funcs.sim_exists(filename)
    if sim_exists:
        return True
    return False


@callback(
    # Output("simulation-run-button", "disabled"),
    # Input("config-make-text", "value"),
    Output({"type": "config-input", "index": MATCH}, "className"),
    Input({"type": "config-input", "index": MATCH}, "value"),
    State({"type": "config-input", "index": MATCH}, "className"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def block_config_input(value, className):
    print(className, "a")
    if ctx.triggered_id is None:
        return className

    param = config.params[ctx.triggered_id["index"]]

    className = className.replace(" disabled", "")

    inside_range = param.resrange(param.convert(value))
    if not inside_range:
        className += " disabled"

    print(className, "a")
    return className

    # for k, value in zip(DEFAULT_CONFIG_DICT, values):
    #     if value != "" and value is not None:
    #         param = config.params[k]
    #         val = param.convert(value)
    #         valid = param.resrange(val)
    #         if not valid:
    #             return True

    # if filename is None or filename == "" or "." in filename:
    #     return True

    # sim_exists = funcs.sim_exists(filename)
    # if sim_exists:
    #     return True
    # return False
