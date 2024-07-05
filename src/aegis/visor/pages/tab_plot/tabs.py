from dash import Dash, dcc, html, Input, Output, callback
from aegis.visor.pages.tab_plot.prep_setup import FIG_SETUP, needs_slider


def get_graph(figure_id):
    info = FIG_SETUP[figure_id]
    return [
        dcc.Graph(
            # id=figure_id,
            id={"type": "graph-figure", "index": figure_id},
            config={"displayModeBar": False},
            className="figure",
        ),
        (
            dcc.Slider(
                min=0,
                max=1,
                step=1,
                value=0,
                marks=None,
                tooltip={},
                id={"type": "graph-slider", "index": figure_id},
            )
            if needs_slider(figure_id)
            else None
        ),
        html.P(
            children=info["title"],
            className="figure-title",
        ),
        html.P(
            children=info["description"],
            className="figure-description",
        ),
        html.Button(
            "download figure",
            id={"type": "figure-download-button", "index": figure_id},
        ),
        dcc.Download(id={"type": "figure-dcc-download", "index": figure_id}),
        # dcc.Slider(0,100)
    ]


def get_tabs_layout():
    tabs = []
    for graph_name in FIG_SETUP:
        children = get_graph(graph_name)
        tab = dcc.Tab(label=graph_name, value=graph_name, children=children)
        tabs.append(tab)
    chosen_tab = next(iter(FIG_SETUP.keys()))
    layout = dcc.Tabs(id="tabs", children=tabs, value=chosen_tab)
    return layout
