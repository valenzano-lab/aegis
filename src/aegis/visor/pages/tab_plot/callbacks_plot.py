import dash
from dash import callback, Output, Input, ctx, ALL, State, MATCH

import logging
from aegis.utilities.container import Container
from aegis.visor.utilities.utilities import get_sim_dir
from aegis.visor.utilities import log_funcs
from aegis.visor.pages.tab_plot import prep_fig
from aegis.visor.pages.tab_plot.prep_setup import FIG_SETUP, needs_slider

from aegis.visor.config import config


@log_funcs.log_debug
def gen_fig(fig_name, selected_sims, containers, iloc):
    """Generates a figure using the figure setup"""

    # Extract setup
    fig_setup = FIG_SETUP[fig_name]

    # Prepare x and y data
    prep_x = fig_setup["prep_x"]
    prep_y = fig_setup["prep_y"]

    ys = []
    max_ilocs = []
    xs = []

    for sim in selected_sims:
        ysi, max_iloc = prep_y(containers[sim], iloc=iloc)
        ys.append(ysi)

        if max_iloc is not None:
            max_iloc -= 1  # before this, it is length
            max_ilocs.append(max_iloc)

        xsi = prep_x(containers[sim], y=ysi)
        xs.append(xsi)

    max_iloc = min(max_ilocs) if max_ilocs else None

    prep_figure = getattr(prep_fig, fig_setup["prep_figure"])
    figure = prep_figure(fig_name, xs, ys, selected_sims)

    return figure, max_iloc


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


# BUG plotting error when dropdown is empty; replace with stylized empty figures
@callback(
    # [Output({"type": "graph-figure", "index": key}, "figure", allow_duplicate=True) for key in FIG_SETUP.keys()],
    Output({"type": "graph-figure", "index": dash.dependencies.ALL}, "figure", allow_duplicate=True),
    Input("dropdown", "value"),
    Input("tabs", "value"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def triggered_dropdown(dropdown_values, tabs_value):
    print(f"plotting {dropdown_values}")

    # BUG apply this logic everywhere. so elegant!
    if not dropdown_values:
        return dash.no_update

    base_dir = get_sim_dir()
    containers = {simname: Container(base_dir / simname) for simname in dropdown_values}

    drag_maxs = []
    figures = []

    for fig_name in FIG_SETUP:
        if fig_name == tabs_value:  # Only update the figure that matches the selected tab
            figure, max_iloc = gen_fig(fig_name, dropdown_values, containers, iloc=-1)
            figures.append(figure)
        else:
            figures.append(dash.no_update)
    return figures


# BUG plot when triggered to plot
@callback(
    [Output({"type": "graph-figure", "index": key}, "figure", allow_duplicate=True) for key in FIG_SETUP.keys()]
    + [Output({"type": "graph-slider", "index": ALL}, "max")],
    # Input("plot-view-button", "n_clicks"),
    Input("reload-plots-button", "n_clicks"),
    # Input("main-url", "pathname"),
    State({"type": "selection-state", "index": ALL}, "data"),
    State({"type": "graph-slider", "index": ALL}, "max"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def update_plot_tab(n_clicks, selection_states, drag_maxs):
    """
    Update plots whenever someone clicks on the plot button or the reload button.
    """
    # If initial call, run the function so that the figures get initialized
    if ctx.triggered_id is None:  # if initial call
        selection_states = list(config.default_selection_states)

    containers = {}
    base_dir = get_sim_dir()
    for filename, selected in selection_states:
        if selected and filename not in containers:
            results_path = base_dir / filename
            logging.info(f"Fetching data from {results_path}.")
            containers[filename] = Container(base_dir / filename)

    selected_sims = [filename for filename, selected in selection_states if selected]
    logging.info("Plotting: " + ", ".join(selected_sims) + ".")

    figures = []
    drag_maxs = []

    for fig_name in FIG_SETUP:
        figure, max_iloc = gen_fig(fig_name, selected_sims, containers, iloc=-1)
        figures.append(figure)
        if needs_slider(fig_name):
            drag_maxs.append(max_iloc)
            assert max_iloc is not None

    # Prepare figures
    # BUG no data saved yet on running simulations or interrupted simulations
    return figures + [drag_maxs]
