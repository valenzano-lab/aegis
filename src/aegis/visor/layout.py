from dash import html, dcc
from aegis.visor.static import FIGURE_INFO
import pathlib

from aegis.parameters import param

# HERE = pathlib.Path(__file__).absolute().parent

app_layout = html.Div(
    id="main-container",
    # className="main-container",
    children=[
        # checkers
        dcc.Interval(id="results-exist-interval", interval=1000, n_intervals=0),
        dcc.Interval(id="process-monitor-interval", interval=1000, n_intervals=0),
        # TITLE SECTION
        html.Div(
            style={"padding": "1rem 0rem 2rem 0"},
            children=[
                html.Div(
                    className="title-section",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flex-wrap": "wrap",
                                "margin-right": "2rem",
                            },
                            children=[
                                html.H1("aegis"),
                                html.Div(
                                    [
                                        html.Img(
                                            src="assets/sim.svg", className="svg-plot"
                                        ),
                                        html.Label("run"),
                                        # html.Button("config view", id="config-view-button"),
                                    ],
                                    id="config-view-button",
                                    className="view-button",
                                ),
                                html.Div(
                                    [
                                        html.Img(
                                            src="assets/list.svg",
                                            className="svg-plot",
                                            style={
                                                "width": "30px",
                                                "height": "34px",
                                                "margin-top": "-2px",
                                            },
                                        ),
                                        html.Label("list"),
                                        # html.Button("result view", id="result-view-button"),
                                    ],
                                    id="result-view-button",
                                    className="view-button",
                                ),
                                html.Div(
                                    [
                                        html.Img(
                                            src="assets/plot.svg", className="svg-plot"
                                        ),
                                        html.Label("plot"),
                                        # html.Button("plot view", id="plot-view-button"),
                                    ],
                                    id="plot-view-button",
                                    className="view-button",
                                ),
                            ],
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    id="sim-section-control",
                                    style={"display": "flex"},
                                    children=[
                                        dcc.Input(
                                            id="config-make-text",
                                            className="control-element",
                                            type="text",
                                            placeholder="unique id",
                                            autoComplete="off",
                                        ),
                                        html.Button(
                                            "run simulation",
                                            id="simulation-run-button",
                                            className="control-element",
                                        ),
                                        html.P("", id="simulation-run-text"),
                                        # html.Button("make config", id="config-make-button"),]
                                    ],
                                ),
                                html.Div(
                                    id="result-section-control",
                                    style={"display": "none"},
                                    children=[],
                                ),
                                html.Div(
                                    id="plot-section-control",
                                    style={
                                        "display": "none",
                                    },
                                    children=[
                                        html.Button(
                                            "reload",
                                            "reload-plots-button",
                                            className="control-element",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # CONFIG SECTION
        html.Div(
            id="sim-section",
            children=[
                html.Div(
                    children=[
                        html.P(
                            [
                                """
                            Using this tab you can customize your simulation and run it.
                            Change the parameter values under the column name VALUE.
                            Run the simulation by giving it a unique id name and clicking the button 'run simulation'.
                            """,
                            ],
                        )
                    ],
                    style={"color": "white"},
                ),
                html.Table(
                    className="config-table",
                    children=[
                        html.Tr(
                            [
                                html.Th("PARAMETER", style={"padding-left": "1.2rem"}),
                                html.Th("VALUE"),
                                html.Th("TYPE"),
                                html.Th("RANGE", className="valid-values"),
                                html.Th("DOMAIN"),
                                html.Th(
                                    "DESCRIPTION", style={"padding-right": "1.2rem"}
                                ),
                            ],
                        )
                    ]
                    + [
                        html.Tr(
                            [
                                html.Td(v.get_name(), style={"padding-left": "1.2rem"}),
                                html.Td(
                                    children=dcc.Input(
                                        type="text",
                                        placeholder=str(v.default)
                                        if v.default is not None
                                        else "",
                                        id=f"config-{k}",
                                        autoComplete="off",
                                    ),
                                ),
                                # html.Td(children=v.dtype.__name__, className=f"dtype-{v.dtype.__name__} dtype"),
                                html.Td(
                                    children=html.Label(
                                        v.dtype.__name__,
                                        className=f"dtype-{v.dtype.__name__} dtype",
                                    )
                                ),
                                html.Td(children=v.drange, className="data-range"),
                                html.Td(
                                    children=html.Label(
                                        v.domain,
                                        className=f"domain-{v.domain} domain",
                                    ),
                                ),
                                html.Td(
                                    v.info,
                                    className="td-info",
                                    style={"padding-right": "0.8rem"},
                                ),
                            ],
                        )
                        for k, v in param.params.items()
                        if not isinstance(v.default, list)
                    ],
                ),
            ],
        ),
        # RESULT SECTION
        html.Div(id="result-section", style={"display": "none"}, children=[]),
        # FIGURE SECTION
        html.Div(
            id="figure-section",
            style={"display": "none"},
            children=[
                html.Div(
                    id="figure-container",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id=figure_id,
                                            config={"displayModeBar": False},
                                            className="figure",
                                        ),
                                    ],
                                    style={"padding-right": "20px"},
                                ),
                                html.Div(
                                    children=[
                                        html.P(
                                            children=info["title"],
                                            className="figure-title",
                                        ),
                                        html.P(
                                            children=info["description"],
                                            className="figure-description",
                                        ),
                                    ]
                                ),
                            ],
                            className="figure-card",
                        )
                        for figure_id, info in FIGURE_INFO.items()
                    ],
                ),
            ]
            # + [
            #     dcc.Graph(id="figurex"),
            #     html.Div(
            #         [
            #             dcc.Slider(
            #                 id="slider",
            #                 min=1,
            #                 max=10,
            #                 step=1,
            #                 value=5,
            #                 updatemode="drag",
            #             ),
            #         ],
            #         # style={"width": "400px"},
            #     ),
            # ],
        ),
        #
        # FOOTER SECTION
        html.Div(
            [
                # html.Hr(),
                # html.A("github link", href="https://github.com/valenzano-lab/aegis", className="footer-text"),
            ]
        ),
    ],
)
