import dash
from aegis_sim.utilities.container import Container
from aegis_gui.utilities import utilities
import dash_bootstrap_components as dbc
from aegis_gui.utilities import log


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

    RECENCY_WINDOW = 365 * 24 * 3600

    for path in paths:
        container = Container(path)
        ticker = container.get_ticker()
        since_last = ticker.since_last()
        if since_last is None:
            continue
        if since_last < RECENCY_WINDOW and container.has_ticker_stopped():
            recent_containers.append(container)

    if recent_containers:
        N_recent_containers_to_show = 20

        buttons = [
            dbc.Button(
                children=[rc.name],
                href=f"/simlog?sim={rc.name}",
                className="badge me-1",
                color="primary",
            )
            for rc in recent_containers[:N_recent_containers_to_show]
        ]

        if len(recent_containers) > N_recent_containers_to_show:
            buttons.append(dbc.Button("more...", className="badge me-1", color="secondary", href="/simlog"))

        nav = dash.html.Div(children=buttons)

        trackers.append(
            dbc.Toast(
                children=nav,
                header="Most recent sims",
                style={"width": "100%", "marginTop": "1rem"},
            )
        )

        # trackers.append(nav)

    return trackers, ticker_store


@log.log_debug
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
