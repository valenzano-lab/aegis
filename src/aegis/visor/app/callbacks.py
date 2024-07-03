from dash import callback, Output, Input, ctx
from aegis.visor.utilities import log_funcs


# @callback(
#     Output("landing-section", "style"),
#     Output("landing-section-control", "style"),
#     Output("plot-section", "style"),
#     Output("plot-section-control", "style"),
#     Output("sim-section", "style"),
#     Output("sim-section-control", "style"),
#     Output("simlog-section", "style"),
#     Output("simlog-section-control", "style"),
#     Input("landing-view-button", "n_clicks"),
#     Input("config-view-button", "n_clicks"),
#     Input("simlog-view-button", "n_clicks"),
#     Input("plot-view-button", "n_clicks"),
#     prevent_initial_call=True,
# )
# @log_funcs.log_info
# def toggle_display(*_):
#     triggered = ctx.triggered_id.split("-")[0]
#     styles = {
#         "landing": [
#             {"display": "flex"},
#             {"display": "flex"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#         ],
#         "plot": [
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "block"},
#             {"display": "block"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#         ],
#         "config": [
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "block"},
#             {"display": "flex"},
#             {"display": "none"},
#             {"display": "none"},
#         ],
#         "list": [
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "none"},
#             {"display": "block"},
#             {"display": "block"},
#         ],
#     }
#     return styles[triggered]
