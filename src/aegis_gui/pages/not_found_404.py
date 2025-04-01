import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = html.Div(
    children=[
    html.H1("Oops!"),
    html.P("This page does not exist. Please navigate back using the menu on the left or the button below."),
    dbc.Button(
        children=[dash.html.I(className="bi bi-house-heart"), "Return home"],
        # id="open-offcanvas-backdrop",
        n_clicks=0,
        outline=True,
        color="primary",
        style={"marginTop": "0.5rem"},
        href='/aegis',
    ),
    ],
    )