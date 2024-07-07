from aegis.visor.pages.tab_plot.prep_setup import FIG_SETUP
from aegis.visor.pages.tab_plot import prep_fig
from aegis.visor.utilities import log_funcs

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