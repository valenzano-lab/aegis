import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from aegis_gui.utilities import sim_tracker


def get_app_layout():
    return html.Div(
        id="body-container",
        children=[
            dcc.Location(id="url", refresh=False),
            get_sidebar(),
            html.Div(id="main-container", children=[dash.page_container]),
        ],
    )


def get_sidebar():
    return dbc.Nav(
        children=[
            dbc.NavItem(
                [dbc.NavLink([dash.html.I(className="bi bi-house-door-fill"), "Home"], href="/", id="link-nav-home")]
            ),
            dbc.NavItem(
                [
                    dbc.NavLink(
                        [dash.html.I(className="bi bi-rocket-takeoff-fill"), "Launch"],
                        href="/config",
                        id="link-nav-config",
                    )
                ]
            ),
            dbc.NavItem(
                [dbc.NavLink([dash.html.I(className="bi bi-bar-chart-fill"), "Plot"], href="/plot", id="link-nav-plot")]
            ),
            dbc.NavItem(
                [
                    dbc.NavLink(
                        [dash.html.I(className="bi bi-eye-fill"), "Control"], href="/simlog", id="link-nav-simlog"
                    )
                ]
            ),
            dbc.NavItem(
                [
                    dbc.NavLink(
                        [dash.html.I(className="bi bi-info-square-fill"), "Wiki"], href="/wiki", id="link-nav-wiki"
                    )
                ]
            ),
        ]
        + sim_tracker.init_tracker_box(),
        id="sidebar",
        vertical="md",
        pills=True,
        # fill=True,
    )