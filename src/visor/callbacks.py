from dash import html, dcc, callback, Output, Input, State, ALL, MATCH, ctx
from visor import funcs
import subprocess


@callback(
    Output("landing-section", "style"),
    Output("landing-section-control", "style"),
    Output("plot-section", "style"),
    Output("plot-section-control", "style"),
    Output("sim-section", "style"),
    Output("sim-section-control", "style"),
    Output("list-section", "style"),
    Output("list-section-control", "style"),
    Input("landing-view-button", "n_clicks"),
    Input("config-view-button", "n_clicks"),
    Input("list-view-button", "n_clicks"),
    Input("plot-view-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.log_info
def toggle_display(*_):
    triggered = ctx.triggered_id.split("-")[0]
    styles = {
        "landing": [
            {"display": "flex"},
            {"display": "flex"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
        ],
        "plot": [
            {"display": "none"},
            {"display": "none"},
            {"display": "flex"},
            {"display": "flex"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
        ],
        "config": [
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "block"},
            {"display": "flex"},
            {"display": "none"},
            {"display": "none"},
        ],
        "list": [
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "block"},
            {"display": "block"},
        ],
    }
    return styles[triggered]
