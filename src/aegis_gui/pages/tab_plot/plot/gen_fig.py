from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot.plot import prep_fig
from aegis_gui.utilities.utilities import get_container


def generate_figure(fig_name, selected_sims, iloc, dark_mode=False):
    """Generates a figure using the figure setup"""

    # Extract setup
    fig_setup = FIG_SETUP[fig_name]

    # Prepare x and y data
    prep_x = fig_setup["prep_x"]
    prep_y = fig_setup["prep_y"]

    ys = []
    max_ilocs = []
    xs = []

    # TODO you might not have to read all the data again when you slide!

    for sim in selected_sims:
        container = get_container(sim)
        ysi, max_iloc = prep_y(container, iloc=iloc)
        ys.append(ysi)

        if max_iloc is not None:
            max_iloc -= 1  # before this, it is length
            max_ilocs.append(max_iloc)

        xsi = prep_x(container, y=ysi)
        xs.append(xsi)

    max_iloc = min(max_ilocs) if max_ilocs else None

    prep_figure = getattr(prep_fig, fig_setup["prep_figure"])
    figure = prep_figure(fig_name, xs, ys, selected_sims, dark_mode=dark_mode)

    return figure, max_iloc
