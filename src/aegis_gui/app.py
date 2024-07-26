import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


nav = dbc.Nav(
    [
        dbc.NavItem(
            [
                html.I(className="bi bi-house-door-fill"),
                dbc.NavLink("home", href="/", id="link-nav-home"),
            ]
        ),
        dbc.NavItem(
            [
                html.I(className="bi bi-gear-fill"),
                dbc.NavLink("config", href="config", id="link-nav-config"),
            ]
        ),
        dbc.NavItem(
            [
                html.I(className="bi bi-bar-chart-fill"),
                dbc.NavLink("plot", href="plot", id="link-nav-plot"),
            ]
        ),
        dbc.NavItem(
            [
                html.I(className="bi bi-list-ul"),
                dbc.NavLink("simlog", href="simlog", id="link-nav-simlog"),
            ]
        ),
        dbc.NavItem(
            [
                html.I(className="bi bi-info-square-fill"),
                dbc.NavLink("wiki", href="wiki", id="link-nav-wiki"),
            ]
        ),
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
        #         html.H1(dbc.NavItem([dbc.NavLink("aegis", href="/"),]),
        #         dbc.NavItem([dbc.NavLink("home", href="/", id="link-nav-home"),
        #         dbc.NavItem([dbc.NavLink("config", href="config", id="link-nav-config"),
        #         dbc.NavItem([dbc.NavLink("plot", href="plot", id="link-nav-plot"),
        #         dbc.NavItem([dbc.NavLink("simlog", href="simlog", id="link-nav-simlog"),
        #         dbc.NavItem([dbc.NavLink("wiki", href="wiki", id="link-nav-wiki"),
        #     ],
        # ),
        nav,
        html.Div(id="main-container", children=[dash.page_container]),
    ],
)


def get_app():
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title="",
        # *.css in assets are automatically imported; they need to be explicitly ignored
        assets_ignore="styles-dark.css",
        use_pages=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    )

    # Bootstrap ICONS: https://icons.getbootstrap.com/

    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout

    return app
