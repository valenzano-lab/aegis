import dash
import dash_bootstrap_components as dbc
from aegis_gui.utilities import sim_tracker

nav = dbc.Nav(
    children=[
        dbc.NavItem(
            [dbc.NavLink([dash.html.I(className="bi bi-house-door-fill"), "Home"], href="/", id="link-nav-home")]
        ),
        dbc.NavItem(
            [
                dbc.NavLink(
                    [dash.html.I(className="bi bi-rocket-takeoff-fill"), "Launch"], href="/config", id="link-nav-config"
                )
            ]
        ),
        dbc.NavItem(
            [dbc.NavLink([dash.html.I(className="bi bi-bar-chart-fill"), "Plot"], href="/plot", id="link-nav-plot")]
        ),
        dbc.NavItem(
            [dbc.NavLink([dash.html.I(className="bi bi-eye-fill"), "Control"], href="/simlog", id="link-nav-simlog")]
        ),
        dbc.NavItem(
            [dbc.NavLink([dash.html.I(className="bi bi-info-square-fill"), "Wiki"], href="/wiki", id="link-nav-wiki")]
        ),
    ]
    + sim_tracker.make_tracker_component(),
    id="sidebar",
    vertical="md",
    pills=True,
    # fill=True,
)
