import dash
from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot import slider
from aegis_gui.pages.tab_plot import download, select_graph


def get_graph(graph_name):
    info = FIG_SETUP[graph_name]
    return dash.html.Div(
        children=[
            dash.dcc.Graph(
                # id=figure_id,
                id={"type": "graph-figure", "index": graph_name},
                config={"displayModeBar": False},
                className="figure",
            ),
            slider.make_slider(graph_name),
            # (slider.make_slider(figure_id) if slider.needs_slider(figure_id) else None),
            dash.html.P(
                children=info["title"],
                className="figure-title",
            ),
            dash.html.P(
                children=info["description"],
                className="figure-description",
            ),
            download.get_figure_download_button(figure_id=graph_name),
            download.get_figure_download_dcc(figure_id=graph_name),
        ],
        id={"type": "graph-div", "index": graph_name},
        className="graph-invisible" if graph_name != select_graph.initial_graph else "",
    )


def get_graphs():
    graph_names = [graph_name for graph_name, d in FIG_SETUP.items() if d["supports_multi"] or not d["supports_multi"]]
    graphs = []
    for graph_name in graph_names:
        graph = get_graph(graph_name)
        graphs.append(graph)
    return graphs
