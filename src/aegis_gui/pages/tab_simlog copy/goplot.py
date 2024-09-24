import dash
from aegis_gui.guisettings.GuiSettings import gui_settings
import dash_bootstrap_components as dbc


def make_button(filename):
    return dash.dcc.Link(
        dbc.Button(
            [dash.html.I(className="bi bi-bar-chart-fill"), "Plot"],
            className="me-2",
            color="primary",
        ),
        href=gui_settings.wrap_href(f"plot?sim={filename}"),
    )
