import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("home", href="/", id="link-nav-home")),
        dbc.NavItem(dbc.NavLink("config", href="config", id="link-nav-config")),
        dbc.NavItem(dbc.NavLink("plot", href="plot", id="link-nav-plot")),
        dbc.NavItem(dbc.NavLink("simlog", href="simlog", id="link-nav-simlog")),
        dbc.NavItem(dbc.NavLink("wiki", href="wiki", id="link-nav-wiki")),
    ],
    id="sidebar",
    vertical="md",
    pills=True,
)

app_layout = html.Div(
    id="body-container",
    children=[
        dcc.Location(id="main-url", refresh=False),
        # checkers
        # dcc.Interval(id="results-exist-interval", interval=1000, n_intervals=0),
        # dcc.Interval(id="process-monitor-interval", interval=1000, n_intervals=0),
        # TITLE SECTION
        # html.Nav(
        #     id="sidebar",
        #     children=[
        #         html.H1(dbc.NavItem(dbc.NavLink("aegis", href="/")),
        #         dbc.NavItem(dbc.NavLink("home", href="/", id="link-nav-home"),
        #         dbc.NavItem(dbc.NavLink("config", href="config", id="link-nav-config"),
        #         dbc.NavItem(dbc.NavLink("plot", href="plot", id="link-nav-plot"),
        #         dbc.NavItem(dbc.NavLink("simlog", href="simlog", id="link-nav-simlog"),
        #         dbc.NavItem(dbc.NavLink("wiki", href="wiki", id="link-nav-wiki"),
        #     ],
        # ),
        nav,
        html.Div(id="main-container", children=[dash.page_container]),
    ],
)
