from dash import html, dcc, callback, Output, Input, State, ALL, MATCH, ctx
from aegis.visor import funcs
import subprocess


@callback(
    Output("figure-section", "style"),
    Output("plot-section-control", "style"),
    Output("sim-section", "style"),
    Output("sim-section-control", "style"),
    Output("list-section", "style"),
    Output("list-section-control", "style"),
    Input("config-view-button", "n_clicks"),
    Input("list-view-button", "n_clicks"),
    Input("plot-view-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.log_debug
def toggle_display(*_):
    triggered = ctx.triggered_id.split("-")[0]
    styles = {
        "plot": [
            {"display": "flex"},
            {"display": "flex"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            # "",
        ],
        "config": [
            {"display": "none"},
            {"display": "none"},
            {"display": "block"},
            {"display": "flex"},
            {"display": "none"},
            {"display": "none"},
            # "bluish-bckg",
        ],
        "list": [
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "block"},
            {"display": "block"},
            # "bluish-bckg",
        ],
    }
    return styles[triggered]
