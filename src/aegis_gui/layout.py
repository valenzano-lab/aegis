import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from aegis_gui.utilities import sim_tracker

# from aegis_gui.utilities import log_funcs


def get_app_layout():
    return html.Div(
        id="body-container",
        children=[
            dcc.Location(id="url", refresh=False),
            get_sidebar(),
            html.Div(id="main-container", children=[dash.page_container]),
        ],
    )


def get_theme_switch():
    return html.Span(
        [
            dbc.Label("Dark mode", html_for="color-mode-switch"),
            dbc.Switch(id="color-mode-switch", value=False, className="d-inline-block ms-1", persistence=True),
        ]
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
        + sim_tracker.init_tracker_box()
        + [get_offcanvas_trigger(), get_offcanvas()],
        id="sidebar",
        vertical="md",
        pills=True,
        # fill=True,
    )


def get_offcanvas_trigger():
    return dbc.Button(
        children=[html.I(className="bi bi-gear-wide"), "Preferences"],
        id="open-offcanvas-backdrop",
        n_clicks=0,
        outline=True,
        color="primary",
        style={"marginTop": "1rem"},
    )


def get_offcanvas():
    children = [get_theme_switch()]
    return dbc.Offcanvas(
        children=children,
        id="offcanvas-backdrop",
        scrollable=True,
        title="User preferences",
    )


@dash.callback(
    dash.Output("offcanvas-backdrop", "is_open"),
    dash.Input("open-offcanvas-backdrop", "n_clicks"),
    dash.State("offcanvas-backdrop", "is_open"),
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


dash.clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'dark' : 'false');  
       return window.dash_clientside.no_update
    }
    """,
    dash.Output("color-mode-switch", "id"),
    dash.Input("color-mode-switch", "value"),
)
