import dash
from dash import callback, Output, Input, ALL, State, dcc

from aegis_gui.utilities import log_funcs
from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot.plot.gen_fig import gen_fig
import dash_bootstrap_components as dbc


# def make_single_dropdown(dropdown_options):
#     # TODO plot initial plots
#     initial_value = dropdown_options[0] if dropdown_options else None
#     initial_value = None
#     return dcc.Dropdown(id="dropdown-single", options=dropdown_options, value=initial_value)


def make_multi_dropdown(dropdown_options, dropdown_value=[]):
    if dropdown_value == []:
        dropdown_value = [dropdown_options[0]] if dropdown_options else []
    assert isinstance(dropdown_value, list)
    return dbc.InputGroup(
        [
            # dbc.InputGroupText("Plotted simulation"),
            dcc.Dropdown(
                id="dropdown-multi",
                options=dropdown_options,
                value=dropdown_value,
                multi=True,
                placeholder="Select simulations to plot...",
                className="plot-dropdown",
                style={"width": "100%"},
            ),
        ]
    )


# Define a helper function to handle the logic
def handle_trigger(dropdown_values, selected_fig):
    if not dropdown_values or dropdown_values == [None]:
        return dash.no_update

    drag_maxs = []
    figures = []

    for fig_name in FIG_SETUP:
        if fig_name == selected_fig:  # Only update the figure that matches the selected tab
            figure, max_iloc = gen_fig(fig_name, dropdown_values, iloc=0)
            figures.append(figure)
            drag_maxs.append(max_iloc)
        else:
            figures.append(dash.no_update)
            drag_maxs.append(dash.no_update)
    return figures, drag_maxs


# TODO take into consideration the existing slider value


# Multi Dropdown and Tabs
@callback(
    Output({"type": "graph-figure", "index": ALL}, "figure"),
    Output({"type": "graph-slider", "index": ALL}, "max"),
    Input("dropdown-multi", "value"),
    Input("figure-select", "value"),
    # Input("tabs-multi", "value"),
    # prevent_initial_call=True,
)
def triggered_dropdown_multi(dropdown_values, figure_selected):
    return handle_trigger(dropdown_values, figure_selected)


@callback(
    Output({"type": "loading-graph-div", "index": ALL}, "parent_style"),
    Input("figure-select", "value"),
    State({"type": "loading-graph-div", "index": ALL}, "id"),
    State({"type": "loading-graph-div", "index": ALL}, "parent_style"),
)
def change_visibility(figure_selected, graph_ids, parent_styles):

    new_parent_styles = []

    for graph_id, parent_style in zip(graph_ids, parent_styles):
        new_parent_style = parent_style.copy() if parent_style else {}
        should_be_invisible = graph_id["index"] != figure_selected

        if should_be_invisible:
            new_parent_style["display"] = "none"
        elif "display" in new_parent_style:
            del new_parent_style["display"]

        new_parent_styles.append(new_parent_style)

    return new_parent_styles
