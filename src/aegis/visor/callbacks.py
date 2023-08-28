from dash import html, dcc, callback, Output, Input, State, ALL, MATCH
import dash

from aegis.visor import funcs


@callback(
    Output("figure-section", "style"),
    Output("sim-section", "style"),
    Output("result-section", "style"),
    Input("plot-view-button", "n_clicks"),
    Input("config-view-button", "n_clicks"),
    Input("result-view-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def toggle_display(*_):
    triggered = dash.callback_context.triggered[0]["prop_id"].split("-")[0]
    styles = {
        "plot": [{"display": "block"}, {"display": "none"}, {"display": "none"}],
        "config": [{"display": "none"}, {"display": "block"}, {"display": "none"}],
        "result": [{"display": "none"}, {"display": "none"}, {"display": "block"}],
    }
    return styles[triggered]


@callback(
    Output("plot-view-button", "disabled"),
    Output("result-view-button", "disabled"),
    Input("results-exist-interval", "n_intervals"),
)
@funcs.print_function_name
def block_view_buttons(_):
    paths = funcs.get_sim_paths()
    if paths:
        return False, False
    else:
        return True, True
