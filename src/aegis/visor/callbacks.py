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

# TODO what is this for
# @callback(
#     Output("plot-view-button", "className"),
#     Output("list-view-button", "className"),
#     Input("results-exist-interval", "n_intervals"),
# )
# def block_view_buttons(_):
#     paths = funcs.get_sim_paths()
#     if paths:
#         return "view-button", "view-button"
#     else:
#         return "view-button disabled", "view-button disabled"


# @callback(
#     Output("reload-plots-button", "style"),
#     Input("process-monitor-interval", "n_intervals"),
#     State("reload-plots-button", "n_clicks"),
# )
# @funcs.log_debug
# def monitor_processes(_, n_clicks):
#     stdout = subprocess.run(
#         ['ps aux | grep "\-\-config_path"'], shell=True, capture_output=True
#     ).stdout.decode()

#     if stdout:
#         lists = [
#             line.split("share/aegis/")[1].strip(".yml")
#             for line in stdout.strip().split("\n")
#         ]
#         sims = [list for list in lists]
#         print(f"Simulations running: {', '.join(sims)}")
#         if sims:
#             if n_clicks is None:
#                 n_clicks = 0
#             else:
#                 n_clicks += 1

