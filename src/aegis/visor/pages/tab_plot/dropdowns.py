import dash
from dash import callback, Output, Input, ctx, ALL, State, MATCH, dcc

from aegis.utilities.container import Container
from aegis.visor.utilities.utilities import get_sim_dir
from aegis.visor.utilities import log_funcs
from aegis.visor.pages.tab_plot.prep_setup import FIG_SETUP
from aegis.visor.pages.tab_plot.gen_fig import gen_fig


def make_single_dropdown(dropdown_options):
    return dcc.Dropdown(id="dropdown-single", options=dropdown_options, value=None)


def make_multi_dropdown(dropdown_options):
    return dcc.Dropdown(id="dropdown-multi", options=dropdown_options, value=None, multi=True)


# Define a helper function to handle the logic
def handle_trigger(dropdown_values, tabs_value, ismulti):
    # BUG apply this logic everywhere. so elegant!
    if not dropdown_values:
        return dash.no_update

    base_dir = get_sim_dir()
    containers = {simname: Container(base_dir / simname) for simname in dropdown_values}

    drag_maxs = []
    figures = []

    for fig_name in FIG_SETUP:
        supports_multi = FIG_SETUP[fig_name]["supports_multi"]
        if fig_name == tabs_value and ismulti == supports_multi:  # Only update the figure that matches the selected tab
            figure, max_iloc = gen_fig(fig_name, dropdown_values, containers, iloc=-1)
            figures.append(figure)
        else:
            figures.append(dash.no_update)
    return figures


# Multi Dropdown and Tabs
@callback(
    Output({"type": "graph-figure", "index": ALL}, "figure", allow_duplicate=True),
    Input("dropdown-multi", "value"),
    Input("tabs-multi", "value"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def triggered_dropdown_multi(dropdown_values, tabs_value):
    return handle_trigger(dropdown_values, tabs_value, ismulti=True)


# Single Dropdown and Tabs
@callback(
    Output({"type": "graph-figure", "index": ALL}, "figure", allow_duplicate=True),
    Input("dropdown-single", "value"),
    Input("tabs-single", "value"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def triggered_dropdown_single(dropdown_value, tabs_value):
    return handle_trigger([dropdown_value], tabs_value, ismulti=False)
