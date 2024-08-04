import dash
from aegis_gui.utilities import utilities, log_funcs

import dash_bootstrap_components as dbc


def make_button(filename):
    return dbc.Button(
        [dash.html.I(className="bi bi-stop-circle"), "terminate"],
        id={"type": "config-terminate-button", "index": filename},
        color="warning",
        value=filename,
        className="me-2",
    )


@dash.callback(
    dash.Output({"type": "config-terminate-button", "index": dash.MATCH}, "n_clicks"),
    dash.Input({"type": "config-terminate-button", "index": dash.MATCH}, "n_clicks"),
)
@log_funcs.log_debug
def button_terminate_simulation(n_clicks):
    if n_clicks is None:
        return dash.no_update

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    filename = eval(button_id)["index"]
    utilities.stop_simulation(filename)

    return dash.no_update
