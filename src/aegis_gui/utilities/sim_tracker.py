import dash
from aegis_sim.utilities.container import Container
from aegis_gui.utilities import utilities
import dash_bootstrap_components as dbc


def make_trackers():
    trackers = []

    running_sims = []

    paths = utilities.get_sim_paths()
    for path in paths:
        container = Container(path)
        is_running = not container.has_ticker_stopped()
        if is_running:
            # tracker = dash.dcc.Link(
            #     children=container.name, href=f"/simlog?sim={container.name}", className="sidebar-sim-running"
            # )
            # tracker = dbc.Button(
            #     children=[container.name],
            #     href=f"/simlog?sim={container.name}",
            #     className="badge me-1 sidebar-sim-running",
            #     color="primary",
            # )
            progress = container.get_simple_log()
            if progress is None:
                continue
            progress = progress[0] / progress[1]
            progressbar = dbc.Progress(
                label=container.name,
                value=10 + progress * 90,
                striped=True,
                animated=True,
                style={"margin-top": "0.5rem"},
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
                style={"width": "100%", "margin-top": "1rem"},
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
                style={"width": "100%", "margin-top": "1rem"},
            )
        )

        # trackers.append(nav)

    return trackers


def get_tracker():
    return [
        dash.dcc.Interval(id="running-simulations-interval", interval=1 * 1000, n_intervals=0),
        dash.html.Div([], id="running-simulations"),
    ]


@dash.callback(
    dash.Output("running-simulations", "children"),
    dash.Input("running-simulations-interval", "n_intervals"),
)
def update_simulations(n):
    return make_trackers()
