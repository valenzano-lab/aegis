import dash
from dash import callback, Output, Input, ctx, ALL, State, MATCH, dcc

from aegis.utilities.container import Container
from aegis_gui.utilities.utilities import get_sim_dir
from aegis_gui.utilities import log_funcs
from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot.plot.gen_fig import gen_fig


# def make_single_dropdown(dropdown_options):
#     # TODO plot initial plots
#     initial_value = dropdown_options[0] if dropdown_options else None
#     initial_value = None
#     return dcc.Dropdown(id="dropdown-single", options=dropdown_options, value=initial_value)


def make_multi_dropdown(dropdown_options):
    # TODO plot initial plots
    return dcc.Dropdown(
        id="dropdown-multi",
        options=dropdown_options,
        value="default" if "default" in dropdown_options else None,
        multi=True,
        className="bootstrap-dropdown",
    )


# Define a helper function to handle the logic
def handle_trigger(dropdown_values, selected_fig):
    # TODO apply this logic everywhere. so elegant!
    if not dropdown_values or dropdown_values == [None]:
        return dash.no_update

    drag_maxs = []
    figures = []

    for fig_name in FIG_SETUP:
        if fig_name == selected_fig:  # Only update the figure that matches the selected tab
            figure, max_iloc = gen_fig(fig_name, dropdown_values, iloc=-1)
            figures.append(figure)
            drag_maxs.append(max_iloc)
        else:
            figures.append(dash.no_update)
            drag_maxs.append(dash.no_update)
    return figures, drag_maxs


# TODO take into consideration the existing slider value


# Multi Dropdown and Tabs
@callback(
    Output({"type": "graph-figure", "index": ALL}, "figure", allow_duplicate=True),
    Output({"type": "graph-slider", "index": ALL}, "max", allow_duplicate=True),
    Input("dropdown-multi", "value"),
    Input("figure-select", "value"),
    # Input("tabs-multi", "value"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def triggered_dropdown_multi(dropdown_values, figure_selected):
    return handle_trigger(dropdown_values, figure_selected)


# # Single Dropdown and Tabs
# @callback(
#     Output({"type": "graph-figure", "index": ALL}, "figure", allow_duplicate=True),
#     Output({"type": "graph-slider", "index": ALL}, "max", allow_duplicate=True),
#     Input("dropdown-single", "value"),
#     Input("figure-select", "value"),
#     # Input("tabs-single", "value"),
#     prevent_initial_call=True,
# )
# @log_funcs.log_debug
# def triggered_dropdown_single(dropdown_value, figure_selected):
#     print(figure_selected)
#     print(dropdown_value)
#     return handle_trigger([dropdown_value], figure_selected, dropdown_multi_triggered=False)


@callback(
    Output({"type": "graph-div", "index": ALL}, "className", allow_duplicate=True),
    Input("figure-select", "value"),
    State({"type": "graph-div", "index": ALL}, "className"),
    State({"type": "graph-div", "index": ALL}, "id"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def change_visibility(figure_selected, class_names, graph_ids):
    new = []
    for class_name, graph_id in zip(class_names, graph_ids):
        class_name_without_invisible = " ".join([c for c in class_name.split() if c != "graph-invisible"])
        should_be_invisible = graph_id["index"] != figure_selected
        if should_be_invisible:
            class_name_without_invisible += " graph-invisible"
        new.append(class_name_without_invisible)
    print(new)
    return new
