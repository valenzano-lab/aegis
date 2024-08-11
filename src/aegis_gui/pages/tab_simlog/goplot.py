import dash
import yaml
from aegis_gui.utilities import log_funcs
from aegis.utilities.container import Container

import dash_bootstrap_components as dbc


def make_button(filename):
    return dash.dcc.Link(
        dbc.Button(
            [dash.html.I(className="bi bi-bar-chart-fill"), "Plot"],
            className="me-2",
            color="primary",
        ),
        href=f"/plot?sim={filename}",
    )
