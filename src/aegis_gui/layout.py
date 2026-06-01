import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from aegis_gui.utilities import sim_tracker
from aegis_gui.utilities.utilities import get_icon
from aegis_gui import offcanvas
from aegis_gui.guisettings.GuiSettings import gui_settings
from aegis_sim import _version as aegis_version


def get_app_layout():
    # Responsive Bootstrap grid: sidebar (3 cols on md+, full width below)
    # next to main (9 cols on md+). On small screens the cols stack — the
    # sidebar appears above main as a horizontal nav strip.
    return dbc.Container(
        id="body-container",
        fluid=True,
        children=[
            dcc.Location(id="url", refresh=False),
            dbc.Row(
                [
                    dbc.Col(get_sidebar(), xs=12, md=3, lg=2, id="sidebar-col"),
                    dbc.Col(
                        html.Div(id="main-container", children=[dash.page_container]),
                        xs=12,
                        md=9,
                        lg=10,
                    ),
                ],
                className="g-0",
            ),
            get_footer(),
        ],
    )


def get_footer():
    """Small version footer at the bottom of every page.

    Shows the running package version, the git commit short hash (when set
    via AEGIS_GIT_COMMIT at container build time), and the build date.
    Lets a sysadmin verify which AEGIS revision is live without inspecting
    the container internals.
    """
    return html.Footer(
        html.Small(
            aegis_version.version_string(),
            className="text-muted",
        ),
        id="aegis-version-footer",
        style={
            "textAlign": "center",
            "padding": "1rem 0 0.5rem 0",
            "marginTop": "2rem",
            "fontSize": "0.75rem",
            "opacity": "0.6",
        },
    )


def get_sidebar():
    return dash.html.Div(
        [
            dbc.Nav(
                children=[
                    html.A(
                        [
                            html.Img(
                                src="assets/aegis-ager.svg", width="80%", style={"margin": "0 10%"}, id="aegis-logo"
                            )
                        ],
                        href=gui_settings.wrap_href(""),
                        className="mb-5",
                    ),
                    dbc.NavItem(
                        [
                            dbc.NavLink(
                                [
                                    dash.html.I(className="bi bi-house-door-fill"),
                                    "Home",
                                    # get_icon("house-door-fill"),
                                ],
                                href=gui_settings.wrap_href(""),
                                id="link-nav-home",
                            )
                        ]
                    ),
                    dbc.NavItem(
                        [
                            dbc.NavLink(
                                [dash.html.I(className="bi bi-rocket-takeoff-fill"), "Launch"],
                                href=gui_settings.wrap_href("config"),
                                id="link-nav-config",
                            )
                        ]
                    ),
                    dbc.NavItem(
                        [
                            dbc.NavLink(
                                [dash.html.I(className="bi bi-bar-chart-fill"), "Plot"],
                                href=gui_settings.wrap_href("plot"),
                                id="link-nav-plot",
                            )
                        ]
                    ),
                    dbc.NavItem(
                        [
                            dbc.NavLink(
                                [dash.html.I(className="bi bi-eye-fill"), "Control"],
                                href=gui_settings.wrap_href("simlog"),
                                id="link-nav-simlog",
                            )
                        ]
                    ),
                    dbc.NavItem(
                        [
                            dbc.NavLink(
                                [dash.html.I(className="bi bi-info-square-fill"), "Wiki"],
                                href=gui_settings.wrap_href("wiki"),
                                id="link-nav-wiki",
                            )
                        ]
                    ),
                    dbc.NavItem(
                        [
                            dbc.NavLink(
                                [dash.html.I(className="bi bi-newspaper"), "News"],
                                href=gui_settings.wrap_href("news"),
                                id="link-nav-news",
                            )
                        ]
                    ),
                ]
                + sim_tracker.init_tracker_box()
                + [offcanvas.get_offcanvas_trigger(), offcanvas.get_offcanvas()],
                id="sidebar",
                vertical="md",
                pills=True,  # TODO fix because it is not showing since i changed the href logic with gui_settings.wrap_href
                # fill=True,
            ),
        ],
    )
