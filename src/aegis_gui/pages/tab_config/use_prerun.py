import dash
import dash_bootstrap_components as dbc
from aegis_gui.utilities import utilities
import logging


def make_select(selected=None):
    return dash.html.Div(
        # id="sim-prerun-control",
        children=[
            dash.html.H6(
                [
                    "Start with a prerun simulation",
                    dbc.Switch(
                        # [dash.html.I(className="bi bi-arrow-up-square-fill"), "Use"],
                        id="prerun-enable-button",
                        className="me-1",
                        value=False,
                        # outline=True,
                        # color="primary",
                        style={"display": "inline-block", "marginLeft": "0.8rem"},
                    ),
                ],
            ),
            dash.html.Div(
                children=[
                    dbc.InputGroup(
                        children=[
                            dbc.InputGroupText("Simulation ID"),
                            dbc.Select(
                                id="prerun-sim-select",
                                # options=[{"label": path.stem, "value": str(path)} for path in paths],
                                value=None,
                                # placeholder="Choose an evolved population",
                                className="me-2",
                                disabled=True,
                            ),
                        ],
                    ),
                ],
                style={"marginBottom": "1rem", "display": "inline-block"},
            ),
        ],
    )


@dash.callback(
    dash.Output("prerun-sim-select", "disabled"),
    dash.Output("prerun-sim-select", "value"),
    dash.Output("prerun-sim-select", "options"),
    dash.Input("prerun-enable-button", "value"),
    # dash.State("prerun-sim-select", "options"),
)
def toggle_select(disable_switch):
    """Enable or disable the dbc.Select based on the dbc.Switch state."""
    if dash.ctx.triggered_id is None:
        return dash.no_update, dash.no_update, dash.no_update

    paths = utilities.get_sim_paths()
    # If no simulations available
    if not paths:
        return True, dash.no_update, dash.no_update

    options = [{"label": path.stem, "value": str(path)} for path in paths]

    new_value = options[0]["value"] if disable_switch else None
    logging.debug(f"Toggle select {new_value}.")
    return not disable_switch, new_value, options
