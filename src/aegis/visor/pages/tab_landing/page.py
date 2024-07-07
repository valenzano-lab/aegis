import dash

from dash import html, dcc
from aegis.visor.utilities import log_funcs
from aegis.documentation.documenter import Documenter

dash.register_page(__name__, name="landing", path="/")


@log_funcs.log_debug
def layout():
    return html.Div(
        id="landing-section",
        children=[
            dcc.Markdown(Documenter.read("1 welcome.md")),
            dcc.Markdown(Documenter.read("1 getting started.md")),
            dcc.Markdown(Documenter.read("1 who is aegis for.md")),
            dcc.Markdown(Documenter.read("1 gallery.md")),
            dcc.Markdown(Documenter.read("2 odd.md")),
            dcc.Markdown(Documenter.read("3 user guide.md")),
            dcc.Markdown(Documenter.read("4 about us and community.md")),
        ],
    )
