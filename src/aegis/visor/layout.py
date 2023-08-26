from dash import html, dcc
from aegis.visor.static import FIGURE_INFO


from aegis.parameters import param

app_layout = html.Div(
    [
        # checkers
        dcc.Interval(id="results-exist-interval", interval=1000, n_intervals=0),

        # TITLE SECTION
        html.Div(
            className="title-section",
            children=[html.H1("AEGIS visor")],
        ),
        # CONTROL SECTION
        html.Div(
            className="control-section",
            children=[
                html.Button("plot view", id="plot-view-button"),
                html.Button("config view", id="config-view-button"),
                html.Button("result view", id="result-view-button"),
            ],
        ),
        # CONFIG SECTION
        html.Div(
            id="sim-section",
            children=[
                html.Button("run simulation", id="simulation-run-button"),
                dcc.Input(
                    id="config-make-text", type="text", placeholder="unique id"
                ),
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
                                html.Th("VALID VALUES"),
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
                                html.Td(children=v.drange, className="data-range"),
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
                # html.Div(
                #     className="dataset-section",
                #     children=[
                #         html.Button(
                #             "reload list of simulations",
                #             "load-paths-button",
                #             style={"display": "none"},
                #         ),
                #         dcc.Dropdown(
                #             id="dynamic-dropdown",
                #             clearable=False,
                #             style={"width": "50%"},
                #         ),
                #     ],
                # ),
            ]
            + [
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
                                    children=info["title"], className="figure-title"
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
            ]
            + [
                dcc.Graph(id="figurex"),
                html.Div(
                    [
                        dcc.Slider(
                            id="slider",
                            min=1,
                            max=10,
                            step=1,
                            value=5,
                            updatemode="drag",
                        ),
                    ],
                    # style={"width": "400px"},
                ),
            ],
        ),
        #
        # FOOTER SECTION
        html.Div(
            [html.Hr(), html.P(children="https://github.com/valenzano-lab/aegis")]
        ),
    ],
    className="main-container",
)
