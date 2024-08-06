import dash
from aegis.utilities.container import Container
from aegis_gui.utilities import utilities
import dash_bootstrap_components as dbc


def make_trackers():
    trackers = []
    paths = utilities.get_sim_paths()
    for path in paths:
        container = Container(path)
        is_running = not container.has_ticker_stopped()
        if is_running:
            tracker = dash.dcc.Link(
                children=container.name, href=f"/simlog?sim={container.name}", className="sidebar-sim-running"
            )
            progress = container.get_simple_log()
            if progress is None:
                continue
            progress = progress[0] / progress[1]
            progressbar = dbc.Progress(
                label=f"{progress*100:.0f}%" if progress > 0.1 else "",
                value=progress * 100,
                striped=True,
                animated=True,
            )
            trackers.append(tracker)
            trackers.append(progressbar)

    trackers.append("Recent simulations:")
    for path in paths:
        container = Container(path)
        ticker = container.get_ticker()
        if ticker.since_last() < 3600 and container.has_ticker_stopped():
            trackers.append(
                dash.dcc.Link(
                    children=container.name, href=f"/simlog?sim={container.name}", className="sidebar-sim-recent"
                )
            )

    return trackers


def make_tracker_component():
    return [
        dash.dcc.Interval(id="running-simulations-interval", interval=1 * 1000, n_intervals=0),
        dash.html.Div(["running-simulations"] + make_trackers(), id="running-simulations"),
    ]


@dash.callback(
    dash.Output("running-simulations", "children"),
    dash.Input("running-simulations-interval", "n_intervals"),
)
def update_simulations(n):
    # Call make_trackers() to get the updated list of elements
    return ["running-simulations"] + make_trackers()
