import dash
from dash import dcc, ALL, Input, callback, Output, MATCH, State
from aegis.visor.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis.visor.pages.tab_plot.plot import prep_x, gen_fig
from aegis.visor.utilities import log_funcs


def make_slider(figure_id):
    return dcc.Slider(
        min=0,
        max=1,
        step=1,
        value=0,
        marks=None,
        tooltip={},
        id={"type": "graph-slider", "index": figure_id},
        className="" if needs_slider(figure_id) else "invisible",
    )


def needs_slider(fig_name):
    if fig_name == "causes of death":
        return False
    return FIG_SETUP[fig_name]["prep_x"] == prep_x.get_ages


# # Single Dropdown and Tabs
# @callback(
#     Output({"type": "graph-figure", "index": MATCH}, "figure"),
#     Input({"type": "graph-slider", "index": MATCH}, "value"),
#     State({"type": "graph-slider", "index": MATCH}, "id"),
#     State("dropdown-multi", "value"),
#     State("dropdown-single", "value"),
#     prevent_initial_call=True,
# )
# @log_funcs.log_debug
# def update(slider_value, slider_id, dropdown_multi_value, dropdown_single_value):

#     fig_name = slider_id["index"]

#     supports_multi = FIG_SETUP[fig_name]["supports_multi"]
#     dropdown_values = dropdown_multi_value if supports_multi else [dropdown_single_value]

#     fig, max_iloc = gen_fig.gen_fig(fig_name, dropdown_values, iloc=-1)

#     return dash.no_update
