import dash
from aegis_gui.utilities import utilities, log_funcs

import dash_bootstrap_components as dbc


def make_button(filename, disabled=False):
    return dbc.Button(
        [dash.html.I(className="bi bi-stop-circle"), "Terminate"],
        id={"type": "config-terminate-button", "index": filename},
        color="warning",
        value=filename,
        className="me-2",
        disabled=disabled,
        # TODO add disabled if there is nothing to terminate
    )


@dash.callback(
    dash.Output({"type": "config-terminate-button", "index": dash.MATCH}, "n_clicks"),
    dash.Input({"type": "config-terminate-button", "index": dash.MATCH}, "n_clicks"),
)

def button_terminate_simulation(n_clicks):
    if n_clicks is None:
        return dash.no_update

    button_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    filename = eval(button_id)["index"]
    # TODO do not use eval
    utilities.terminate_simulation(filename)
    return dash.no_update
