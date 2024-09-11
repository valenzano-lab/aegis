import dash
import dash_bootstrap_components as dbc
from aegis_gui.utilities import log_funcs

seconds_per_refresh = 10
progress_granularity = 0.01  # 1% of the progress bar
seconds_per_progress_step = seconds_per_refresh * progress_granularity


inputgroup = dash.html.Div(
    [
        dbc.Button(
            [dash.html.I(className="bi bi-arrow-clockwise"), "Refresh"],
            id="refresh-figure-data",
            className="me-1",
            outline=True,
            color="secondary",
        ),
        dbc.Button(
            [dash.html.I(className="bi bi-arrow-repeat"), "Keep refreshing"],
            id="keep-refresh-figure-data",
            outline=True,
            color="secondary",
            className="me-1",
            active=True,
        ),
        dash.html.Div(
            children=[
                dbc.Progress(
                    id="refresh-progress-bar",
                    value=0,
                    # striped=True,
                    # animated=True,
                    style={
                        # "height": "8px",
                        "width": "38px",
                        "transform": "rotate(-90deg)",
                        "transformOrigin": "top left",
                        "translate": "0 38px",
                    },
                )
            ],
            className="me-1",
            style={
                # "height": "8px",
                # "width": "38px",
            },
        ),
        dash.dcc.Interval(
            id="progress-interval-refresh",
            interval=1000 * seconds_per_progress_step,
            n_intervals=0,
            disabled=False,
        ),
        dash.dcc.Store(
            id="refresh-progress-store",
            data={"progress": 0},
        ),
    ],
    style={"display": "flex"},
)

# TODO probably unify these two because the progressbar is not going to 0 sometimes when button is unactived


@dash.callback(
    dash.Output("progress-interval-refresh", "disabled"),
    dash.Output("keep-refresh-figure-data", "active"),
    dash.Output("refresh-progress-store", "data", allow_duplicate=True),
    dash.Input("keep-refresh-figure-data", "n_clicks"),
    dash.State("progress-interval-refresh", "disabled"),
    dash.State("refresh-progress-store", "data"),
    prevent_initial_call=True,
)
@log_funcs.log_info
def toggle_interval(n_clicks, is_disabled, progress_data):
    if n_clicks is None:
        raise dash.no_update

    if is_disabled:
        # If the interval is disabled, enable it and reset progress
        return False, True, {"progress": 0}
    else:
        # Otherwise, disable the interval and stop progress
        return True, False, {"progress": 0}


@dash.callback(
    dash.Output("refresh-progress-bar", "value"),
    dash.Output("refresh-progress-store", "data", allow_duplicate=True),
    dash.Output("refresh-figure-data", "n_clicks"),
    dash.Input("progress-interval-refresh", "n_intervals"),
    dash.State("refresh-progress-store", "data"),
    dash.State("refresh-figure-data", "n_clicks"),
    prevent_initial_call=True,
)
# @log_funcs.log_info
def update_progress(n_intervals, progress_data, n_clicks):
    progress = progress_data["progress"]

    # Increment the progress by 10% each second (adjust as needed)
    progress += progress_granularity * 100

    if progress > 100:
        return 0, {"progress": 0}, 1 if n_clicks is None else n_clicks + 1
    else:
        return progress, {"progress": progress}, dash.no_update
