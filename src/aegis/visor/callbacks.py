from dash import Dash, html, dcc, callback, Output, Input, State

from aegis.visor import funcs


@callback(
    Output("sim-section", "style"),
    Output("figure-section", "style"),
    Input("toggle-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def toggle_display(n_clicks):
    div1_style = {"display": "block"}
    div2_style = {"display": "none"}

    if n_clicks % 2 == 1:
        div1_style, div2_style = div2_style, div1_style

    return div1_style, div2_style


@callback(
    [
        Output("dynamic-dropdown", "options"),
        Output("dynamic-dropdown", "value"),
    ],
    [
        Input("toggle-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def refresh_dropdown_options(_):
    paths = [p for p in funcs.BASE_DIR.iterdir() if p.is_dir()]
    dropdown_options = [
        {"label": f"{i}. {str(path.stem)} ({str(path)})", "value": str(path)}
        for i, path in enumerate(paths)
    ]
    return dropdown_options, dropdown_options[0]["value"]


@callback(
    Output("simulation-run-button", "style"),
    Input("simulation-run-button", "n_clicks"),
    State("config-make-text", "value"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def run_simulation(n_clicks, filename):
    if n_clicks is not None:
        funcs.run(filename)


@callback(
    Output("config-make-button", "style"),
    Input("config-make-button", "n_clicks"),
    State("config-make-text", "value"),
    [
        State(f"config-{k}", "value")
        for k, v in funcs.DEFAULT_CONFIG_DICT.items()
        if not isinstance(v, list)
    ],
    prevent_initial_call=True,
)
@funcs.print_function_name
def make_config_file(n_clicks, filename, *values):

    custom_config = {
        k: val
        for (k, v), val in zip(funcs.DEFAULT_CONFIG_DICT.items(), values)
        if not isinstance(v, list)
    }

    if n_clicks is not None:
        funcs.make_config_file(filename, custom_config)
    return {}
