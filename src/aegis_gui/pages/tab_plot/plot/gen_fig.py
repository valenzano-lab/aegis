from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot.plot import prep_fig
from aegis_gui.utilities import log_funcs
from aegis.utilities.container import Container
from aegis_gui.utilities.utilities import get_sim_dir


def get_container(sim_name):
    base_dir = get_sim_dir()
    container = Container(base_dir / sim_name)
    return container


@log_funcs.log_debug
def gen_fig(fig_name, selected_sims, iloc):
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
    figure = prep_figure(fig_name, xs, ys, selected_sims)

    return figure, max_iloc
