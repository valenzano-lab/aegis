from dash import callback, Output, Input, ctx, ALL, State, MATCH, html
from aegis.visor.pages.tab_plot.prep_setup import FIG_SETUP, needs_slider
from aegis.visor.config import config
from aegis.visor.utilities.utilities import get_sim_dir
from aegis.visor.pages.tab_plot.gen_fig import gen_fig
from aegis.utilities.container import Container
from aegis.visor.utilities import log_funcs
import logging


def make_reload_button():
    return html.Button(
        "reload",
        "reload-plots-button",
        className="control-element",
    )


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
