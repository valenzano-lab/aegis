import dash
from dash import html, dcc

from . import run_simulation_button

app_layout = html.Div(
    id="main-container",
    children=[
        dcc.Location(id="main-url", refresh=False),
        # checkers
        # dcc.Interval(id="results-exist-interval", interval=1000, n_intervals=0),
        # dcc.Interval(id="process-monitor-interval", interval=1000, n_intervals=0),
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
                                # html.Div(
                                #     [
                                #         html.Img(src="assets/sim.svg", className="svg-plot"),
                                #         html.Label("home"),
                                #         # html.Button("config view", id="config-view-button"),
                                #     ],
                                #     id="landing-view-button",
                                #     className="view-button",
                                # ),
                                # html.Div(
                                #     [
                                #         html.Img(src="assets/sim.svg", className="svg-plot"),
                                #         html.Label("run"),
                                #         # html.Button("config view", id="config-view-button"),
                                #     ],
                                #     id="config-view-button",
                                #     className="view-button",
                                # ),
                                # html.Div(
                                #     [
                                #         html.Img(
                                #             src="assets/list.svg",
                                #             className="svg-plot",
                                #             style={
                                #                 "width": "30px",
                                #                 "height": "34px",
                                #                 "margin-top": "-2px",
                                #             },
                                #         ),
                                #         html.Label("list"),
                                #         # html.Button("list view", id="simlog-view-button"),
                                #     ],
                                #     id="simlog-view-button",
                                #     className="view-button",
                                # ),
                                # html.Div(
                                #     [
                                #         html.Img(src="assets/plot.svg", className="svg-plot"),
                                #         html.Label("plot"),
                                #         # html.Button("plot view", id="plot-view-button"),
                                #     ],
                                #     id="plot-view-button",
                                #     className="view-button",
                                # ),
                            ],
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    id="landing-section-control",
                                    style={"display": "none"},
                                    children=[],
                                ),
                                run_simulation_button.layout,
                                html.Div(
                                    id="simlog-section-control",
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
        html.Div([dcc.Link(page["name"] + " ", href=page["path"]) for page in dash.page_registry.values()]),
        dash.page_container,
        html.Hr(),
        # TABS
        # get_landing_layout(),
        # get_config_layout(),
        # get_list_layout(),
        # get_plot_layout(),
        # FOOTER SECTION
        html.Div(
            [
                # html.Hr(),
                # html.A("github link", href="https://github.com/valenzano-lab/aegis", className="footer-text"),
            ]
        ),
    ],
)
