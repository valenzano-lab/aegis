from dash import html
from visor import utilities


@utilities.log_debug
def get_landing_layout():
    return html.Div(
        id="landing-section",
        children=["landing"],
    )
