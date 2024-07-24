import dash
from dash import html, dcc


app_layout = html.Div(
    id="body-container",
    children=[
        dcc.Location(id="main-url", refresh=False),
        # checkers
        # dcc.Interval(id="results-exist-interval", interval=1000, n_intervals=0),
        # dcc.Interval(id="process-monitor-interval", interval=1000, n_intervals=0),
        # TITLE SECTION
        html.Nav(
            id="sidebar",
            children=[
                html.H1(dcc.Link("aegis", href="/")),
                dcc.Link("home", href="/", id="link-nav-home", className="nav-item"),
                dcc.Link("config", href="config", id="link-nav-config", className="nav-item"),
                dcc.Link("plot", href="plot", id="link-nav-plot", className="nav-item"),
                dcc.Link("simlog", href="simlog", id="link-nav-simlog", className="nav-item"),
                dcc.Link("wiki", href="wiki", id="link-nav-wiki", className="nav-item"),
            ],
        ),
        html.Div(id="main-container", children=[dash.page_container]),
    ],
)
