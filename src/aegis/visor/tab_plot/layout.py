from dash import html, dcc
from aegis.visor.tab_plot.static import FIGURE_INFO


def get_plot_layout():
    return html.Div(
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
    )
