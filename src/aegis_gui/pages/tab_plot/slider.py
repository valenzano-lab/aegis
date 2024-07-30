import dash
from dash import dcc, ALL, Input, callback, Output, MATCH, State
from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot.plot import prep_x, gen_fig
from aegis_gui.utilities import log_funcs


def make_slider(figure_id):
    return dcc.Slider(
        id={"type": "graph-slider", "index": figure_id},
        className="" if needs_slider(figure_id) else "invisible",
        min=0,
        max=1,
        step=1,
        value=0,
        marks=None,
        tooltip={},
        updatemode="drag",
    )


def needs_slider(fig_name):
    if fig_name == "death table":
        return False
    return FIG_SETUP[fig_name]["prep_x"] == prep_x.get_ages


# Single Dropdown and Tabs
@callback(
    Output({"type": "graph-figure", "index": MATCH}, "figure", allow_duplicate=True),
    Output({"type": "graph-slider", "index": MATCH}, "max", allow_duplicate=True),  # updates slider max as you use it
    Input({"type": "graph-slider", "index": MATCH}, "value"),
    State({"type": "graph-slider", "index": MATCH}, "id"),
    State("dropdown-multi", "value"),
    # State("dropdown-single", "value"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def update(slider_value, slider_id, dropdown_multi_value):
    fig_name = slider_id["index"]
    # supports_multi = FIG_SETUP[fig_name]["supports_multi"]
    # dropdown_values = dropdown_multi_value if supports_multi else [dropdown_single_value]
    dropdown_values = dropdown_multi_value
    fig, max_iloc = gen_fig.gen_fig(fig_name, dropdown_values, iloc=slider_value)
    return fig, max_iloc
