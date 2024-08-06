import dash
from aegis.utilities.container import Container
from aegis_gui.utilities import utilities


def make_trackers():
    trackers = []
    paths = utilities.get_sim_paths()
    for path in paths:
        container = Container(path)
        is_running = not container.has_ticker_stopped()
        if is_running:
            tracker = dash.html.Div(container.name)
            trackers.append(tracker)

    return trackers


def make_tracker_component():
    return [
        dash.dcc.Interval(id="running-simulations-interval", interval=5 * 1000, n_intervals=0),
        dash.html.Div(["running-simulations"] + make_trackers(), id="running-simulations"),
    ]


@dash.callback(
    dash.Output("running-simulations", "children"),
    dash.Input("running-simulations-interval", "n_intervals"),
)
def update_simulations(n):
    # Call make_trackers() to get the updated list of elements
    print("updating")
    return ["running-simulations"] + make_trackers()
