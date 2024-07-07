import dash
from dash import callback, Output, Input, ctx, ALL, State, MATCH

from aegis.utilities.container import Container
from aegis.visor.utilities.utilities import get_sim_dir
from aegis.visor.utilities import log_funcs
from aegis.visor.pages.tab_plot.prep_setup import FIG_SETUP
from aegis.visor.pages.tab_plot.gen_fig import gen_fig


# @callback(
#     Output({"type": "graph-figure", "index": MATCH}, "figure"),
#     Input({"type": "graph-slider", "index": MATCH}, "drag_value"),
#     State({"type": "selection-state", "index": ALL}, "data"),
# )
# @log_funcs.log_debug
# def update_plot_on_sliding(drag_value, selection_states):
#     if ctx.triggered_id is None:
#         return None

#     fig_name = ctx.triggered_id["index"]

#     containers = {}
#     base_dir = get_sim_dir()
#     for filename, selected in selection_states:
#         if selected and filename not in containers:
#             results_path = base_dir / filename
#             logging.info(f"Fetching data from {results_path}.")
#             containers[filename] = Container(base_dir / filename)

#     selected_sims = [filename for filename, selected in selection_states if selected]
#     logging.info(f"Plotting {fig_name} at t={drag_value} : " + ", ".join(selected_sims) + ".")

#     # Prepare figures
#     # BUG no data saved yet on running simulations or interrupted simulations
#     figure, max_iloc = gen_fig(fig_name, selected_sims, containers, iloc=drag_value)
#     return figure


# @callback(
#     Output({"type": "graph-slider", "index": ALL}, "max"),
#     Input("reload-plots-button", "n_clicks"),
#     State({"type": "graph-slider", "index": ALL}, "max"),
# )
# def update_slider_length(n_clicks, maxs):
#     containers = {"default": Container(get_sim_dir() / "default")}
#     selected_sims = ["default"]

#     new_maxs = []

#     for fig_setup in FIG_SETUP.values():
#         if fig_setup["prep_x"] == prep_x.get_steps:
#             new_max = 100
#         elif fig_setup["prep_x"] == prep_x.get_ages:
#             new_max = 10
#         else:
#             assert False, "Leaky if"

#     return new_maxs

