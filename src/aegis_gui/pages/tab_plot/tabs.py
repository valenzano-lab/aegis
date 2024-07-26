# import dash
# from gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
# from .graph import get_graph


# def get_tabs_multi_layout():
#     tabs = []
#     graph_names = [graph_name for graph_name, d in FIG_SETUP.items() if d["supports_multi"]]
#     for graph_name in graph_names:
#         children = get_graph(graph_name)
#         tab = dash.dcc.Tab(label=graph_name, value=graph_name, children=children)
#         tabs.append(tab)
#     layout = dash.dcc.Tabs(id="tabs-multi", children=tabs, value=graph_names[0])
#     return layout


# def get_tabs_single_layout():
#     tabs = []
#     graph_names = [graph_name for graph_name, d in FIG_SETUP.items() if not d["supports_multi"]]
#     for graph_name in graph_names:
#         children = get_graph(graph_name)
#         tab = dash.dcc.Tab(label=graph_name, value=graph_name, children=children)
#         tabs.append(tab)
#     layout = dash.dcc.Tabs(id="tabs-single", children=tabs, value=graph_names[0])
#     return layout
