import dash
from aegis_sim.utilities.container import Container
from aegis_gui.utilities import utilities
import dash_bootstrap_components as dbc
from aegis_gui.utilities import log_funcs


def make_trackers(ticker_store):
    trackers = []

    running_sims = []
    paths = utilities.get_sim_paths()
    for path in paths:

        ticker_store_status = ticker_store.get(path.stem)
        ticker_registered_and_finished = ticker_store_status is False
        if ticker_registered_and_finished:
            continue

        container = Container(path)
        is_running = not container.has_ticker_stopped()
        ticker_store[path.stem] = is_running
        if is_running:
            progress = container.get_simple_log()
            if progress is None:
                continue
            progress = progress[0] / progress[1]
            progressbar = dbc.Progress(
                label=container.name,
                value=10 + progress * 90,
                striped=True,
                animated=True,
                style={"marginTop": "0.5rem"},
            )
            linked_progressbar = dash.dcc.Link(
                children=progressbar,
                href=f"/simlog?sim={container.name}",
            )
            # running_sims.append(tracker)
            running_sims.append(linked_progressbar)

    if running_sims:
        trackers.append(
            dbc.Toast(
                children=[
                    # dash.html.Hr(),
                    # dash.html.P("Running simulations", className="text-secondary", style={"margin": 0}),
                ]
                + running_sims,
                header="Running sims",
                style={"width": "100%", "marginTop": "1rem"},
            )
        )

    recent_containers = []

    for path in paths:
        container = Container(path)
        ticker = container.get_ticker()
        if ticker.since_last() < 3600 * 24 and container.has_ticker_stopped():
            recent_containers.append(container)

    if recent_containers:
        # trackers.append(dash.html.P("Recent simulations", className="text-secondary"))

        nav = dash.html.Div(
            children=[
                dbc.Button(
                    children=[rc.name],
                    href=f"/simlog?sim={rc.name}",
                    className="badge me-1",
                    color="primary",
                )
                for rc in recent_containers
            ],
        )

        trackers.append(
            dbc.Toast(
                children=nav,
                header="Recent sims",
                style={"width": "100%", "marginTop": "1rem"},
            )
        )

        # trackers.append(nav)

    return trackers, ticker_store


@log_funcs.log_info
def init_tracker_box():
    trackers, ticker_store = get_tracker_box(None, {})
    return [
        dash.dcc.Interval(
            id="running-simulations-interval", interval=2 * 1000, n_intervals=0
        ),  # TODO potential performance issues
        dash.dcc.Store(id="ticker-store", data=ticker_store),
        dash.html.Div(trackers, id="running-simulations"),
    ]


@dash.callback(
    dash.Output("running-simulations", "children"),
    dash.Output("ticker-store", "data"),
    dash.Input("running-simulations-interval", "n_intervals"),
    dash.State("ticker-store", "data"),
)
def get_tracker_box(n, ticker_store):
    # NOTE This takes a long time. If the interval refresh rate is very high, it will be retriggered before finishing and no trackers will appear.
    trackers, ticker_store = make_trackers(ticker_store)
    return trackers, ticker_store
