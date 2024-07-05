import dash
from dash import html, dcc


app_layout = html.Div(
    id="main-container",
    children=[
        dcc.Location(id="main-url", refresh=False),
        # checkers
        # dcc.Interval(id="results-exist-interval", interval=1000, n_intervals=0),
        # dcc.Interval(id="process-monitor-interval", interval=1000, n_intervals=0),
        # TITLE SECTION
        html.Div(
            style={"padding": "1rem 0rem 2rem 0"},
            children=[
                html.Div(
                    className="title-section",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flex-wrap": "wrap",
                                "margin-right": "2rem",
                            },
                            children=[
                                html.H1(dcc.Link("aegis", href="/")),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            [
                dcc.Link("home", href="/", id="link-nav-home"),
                dcc.Link("config", href="config", id="link-nav-config"),
                dcc.Link("plot", href="plot", id="link-nav-plot"),
                dcc.Link("simlog", href="simlog", id="link-nav-simlog"),
                dcc.Link("wiki", href="wiki", id="link-nav-wiki"),
            ],
        ),
        dash.page_container,
        html.Hr(),
        # FOOTER SECTION
        html.Div(
            [
                # html.Hr(),
                # html.A("github link", href="https://github.com/valenzano-lab/aegis", className="footer-text"),
            ]
        ),
    ],
)
