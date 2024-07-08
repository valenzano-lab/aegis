from dash import Dash, dcc, html, Input, Output, callback
from aegis.visor.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis.visor.pages.tab_plot import slider

from aegis.visor.pages.tab_plot import download


def get_graph(figure_id):
    info = FIG_SETUP[figure_id]
    return [
        dcc.Graph(
            # id=figure_id,
            id={"type": "graph-figure", "index": figure_id},
            config={"displayModeBar": False},
            className="figure",
        ),
        (slider.make_slider(figure_id) if slider.needs_slider(figure_id) else None),
        html.P(
            children=info["title"],
            className="figure-title",
        ),
        html.P(
            children=info["description"],
            className="figure-description",
        ),
        download.get_figure_download_button(figure_id=figure_id),
        download.get_figure_download_dcc(figure_id=figure_id),
    ]


def get_tabs_multi_layout():
    tabs = []
    graph_names = [graph_name for graph_name, d in FIG_SETUP.items() if d["supports_multi"]]
    for graph_name in graph_names:
        children = get_graph(graph_name)
        tab = dcc.Tab(label=graph_name, value=graph_name, children=children)
        tabs.append(tab)
    layout = dcc.Tabs(id="tabs-multi", children=tabs, value=graph_names[0])
    return layout


def get_tabs_single_layout():
    tabs = []
    graph_names = [graph_name for graph_name, d in FIG_SETUP.items() if not d["supports_multi"]]
    for graph_name in graph_names:
        children = get_graph(graph_name)
        tab = dcc.Tab(label=graph_name, value=graph_name, children=children)
        tabs.append(tab)
    layout = dcc.Tabs(id="tabs-single", children=tabs, value=graph_names[0])
    return layout
