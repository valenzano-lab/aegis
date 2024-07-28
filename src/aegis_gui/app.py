import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from aegis_gui.utilities import log_funcs


nav = dbc.Nav(
    children=[
        dbc.NavItem([dbc.NavLink([html.I(className="bi bi-house-door-fill"), "Home"], href="/", id="link-nav-home")]),
        dbc.NavItem(
            [dbc.NavLink([html.I(className="bi bi-gear-fill"), "Config"], href="/config", id="link-nav-config")]
        ),
        dbc.NavItem(
            [dbc.NavLink([html.I(className="bi bi-bar-chart-fill"), "Plot"], href="/plot", id="link-nav-plot")]
        ),
        dbc.NavItem([dbc.NavLink([html.I(className="bi bi-list-ul"), "Simlog"], href="/simlog", id="link-nav-simlog")]),
        dbc.NavItem(
            [dbc.NavLink([html.I(className="bi bi-info-square-fill"), "Wiki"], href="/wiki", id="link-nav-wiki")]
        ),
    ],
    id="sidebar",
    vertical="md",
    pills=True,
    # fill=True,
)


@dash.callback(
    [dash.Output(f"link-nav-{page}", "active") for page in ["home", "config", "plot", "simlog", "wiki"]],
    [dash.Input("url", "pathname")],
)
@log_funcs.log_debug
def toggle_active_links(pathname):
    if pathname is None:
        # Default to home if pathname is None
        pathname = "/"
    return [
        pathname == f"/{page}" or (page == "home" and pathname == "/")
        for page in ["home", "config", "plot", "simlog", "wiki"]
    ]


app_layout = html.Div(
    id="body-container",
    children=[
        dcc.Location(id="url", refresh=False),
        # checkers
        # dcc.Interval(id="results-exist-interval", interval=1000, n_intervals=0),
        # dcc.Interval(id="process-monitor-interval", interval=1000, n_intervals=0),
        # TITLE SECTION
        # html.Nav(
        #     id="sidebar",
        #     children=[Item([dbc.NavLink("wiki", href="wiki", id="link-nav-wiki"),
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
