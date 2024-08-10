import dash

from dash import html, dcc
from aegis_gui.utilities import log_funcs
from aegis.documentation.documenter import Documenter
import dash_bootstrap_components as dbc
from aegis_gui.pages.tab_landing.jumbotron import jumbotron

dash.register_page(__name__, name="landing", path="/")


@log_funcs.log_debug
def layout():
    return html.Div(
        id="landing-section",
        children=[
            dcc.Markdown(Documenter.read("1 welcome.md")),
            jumbotron,
            dcc.Markdown(Documenter.read("1 getting started.md")),
            dbc.Card(
                [
                    dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
                    dbc.CardBody(
                        [
                            html.H4("Card title", className="card-title"),
                            html.P(
                                "Some quick example text to build on the card title and "
                                "make up the bulk of the card's content.",
                                className="card-text",
                            ),
                            dbc.Button("Go somewhere", color="primary"),
                        ]
                    ),
                ],
                style={"width": "18rem"},
            ),
            dcc.Markdown(Documenter.read("1 who is aegis for.md")),
            dcc.Markdown(Documenter.read("1 gallery.md")),
            dcc.Markdown(Documenter.read("2 odd.md")),
            dcc.Markdown(Documenter.read("3 user guide.md")),
            dcc.Markdown(Documenter.read("4 about us and community.md")),
        ],
    )
