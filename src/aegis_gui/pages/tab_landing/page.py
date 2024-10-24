import dash

from dash import html, dcc
import dash_bootstrap_components as dbc
from aegis_gui.pages.tab_landing import jumbotron, usecases
from aegis_gui.guisettings.GuiSettings import gui_settings

dash.register_page(__name__, name="AEGIS", path="/")


def layout():
    return (
        dbc.Container(
            [
                # html.H1("AEGIS", style={"font-weight": "800", "font-size": "50px"}),
                html.H1("Home tab"),
                html.P(
                    "AEGIS (Aging of Evolving Genomes In Silico) is an advanced software tool that simulates the evolution of life history traits at the genotype-phenotype level. By modeling factors such as resource availability, mortality, mutation rates, and reproductive strategies, AEGIS offers unparalleled insights into the dynamics of life history evolution."
                ),
                jumbotron.jumbotron,
                html.H2("How It Works", className="my-4"),
                html.P(
                    "AEGIS operates by simulating individual organisms within a shared environment. The tool models various life history traits such as survival, reproduction, and mutation rates under different selective pressures. Users can define parameters, run simulations, and analyze outcomes to gain insights into evolutionary processes."
                ),
                html.H2("Use Cases", className="my-4"),
                html.P(
                    "AEGIS is designed to be a tool for theoretical and applied research, as well as a learning aid."
                ),
                usecases.layout,
                html.H2("Concepts modeled", className="my-2"),
                html.Ul(
                    [
                        html.Li("Population dynamics"),
                        html.Li("Starvation"),
                        html.Li("Infection"),
                        html.Li("Predation"),
                        html.Li("Weather mortality"),
                        html.Li("Environmental drift"),
                        html.Li("Pseudogenetic simulation"),
                    ]
                ),
                # html.H2("Contact Us", className="mb-2 mt-4"),
                html.Hr(className="mt-4"),
                html.Div(
                    html.Div(
                        [
                            dbc.Button(
                                [dash.html.I(className="bi bi-envelope-fill"), "Email"],
                                color="secondary",
                                className="me-1",
                                href="mailto:",
                                external_link=True,
                                # outline=True,
                                # style={"backgroundColor": "#1c9be9", "color": "white"},
                            ),
                            dbc.Button(
                                [dash.html.I(className="bi bi-twitter"), "Twitter"],
                                # color="#1c9be9",
                                className="me-1",
                                href="https://x.com/valenzanolab",
                                external_link=True,
                                # outline=True,
                                style={"backgroundColor": "#1c9be9", "color": "white"},
                            ),
                            dbc.Button(
                                [dash.html.I(className="bi bi-github"), "Github"],
                                color="#0e1017",
                                className="me-1",
                                href="https://github.com/valenzano-lab/aegis",
                                external_link=True,
                                # outline=True,
                                style={"backgroundColor": "#0e1017", "color": "white"},
                            ),
                            dbc.Button(
                                [dash.html.I(className="bi bi-globe2"), "Lab website"],
                                color="primary",
                                className="me-1",
                                href="https://valenzano-lab.github.io/labsite/",
                                external_link=True,
                                # outline=True,
                            ),
                        ]
                    ),
                ),
                # html.P(html.Small(f"Version: ...", className="text-secondary")),
            ]
        ),
    )
