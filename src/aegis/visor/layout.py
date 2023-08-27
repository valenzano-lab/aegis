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
            className="title-section",
            children=[
                html.H1("AEGIS visor"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(src="assets/sim.svg", className="svg-plot"),
                                html.Label("run"),
                                # html.Button("config view", id="config-view-button"),
                            ],
                            id="config-view-button",
                            className="view-button",
                        ),
                        html.Div(
                            [
                                html.Img(src="assets/list.svg", className="svg-plot"),
                                html.Label("list"),
                                # html.Button("result view", id="result-view-button"),
                            ],
                            id="result-view-button",
                            className="view-button",
                        ),
                        html.Div(
                            [
                                html.Img(src="assets/plot.svg", className="svg-plot"),
                                html.Label("plot"),
                                # html.Button("plot view", id="plot-view-button"),
                            ],
                            id="plot-view-button",
                            className="view-button",
                        ),
                    ],
                    style={"display": "flex", "flex-wrap": "wrap"},
                ),
            ],
        ),
        # CONFIG SECTION
        html.Div(
            id="sim-section",
            children=[
                html.Button("run simulation", id="simulation-run-button"),
                dcc.Input(id="config-make-text", type="text", placeholder="unique id"),
                html.P("", id="simulation-run-text"),
                # html.Button("make config", id="config-make-button"),
            ]
            + [
                html.Table(
                    className="config-table",
                    children=[
                        html.Tr(
                            [
                                html.Th("PARAMETER"),
                                html.Th("VALUE"),
                                html.Th("VALID TYPES"),
                                html.Th("VALID VALUES", className="valid-values"),
                                html.Th("PARAMETER TYPE"),
                                html.Th("PARAMETER DESCRIPTION"),
                            ],
                        )
                    ]
                    + [
                        html.Tr(
                            [
                                html.Td(k),
                                html.Td(
                                    children=dcc.Input(
                                        type="text",
                                        placeholder=str(v.default)
                                        if v.default is not None
                                        else "",
                                        id=f"config-{k}",
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
                                html.Td(v.info, className="td-info"),
                            ],
                        )
                        for k, v in param.params.items()
                        if not isinstance(v.default, list)
                    ],
                )
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
                    html.Button(
                        "reload",
                        "reload-plots-button",
                    ),
                    style={"margin": "0.5rem 0 1rem 0"},
                ),
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
                html.P(children="https://github.com/valenzano-lab/aegis"),
            ]
        ),
    ],
)
