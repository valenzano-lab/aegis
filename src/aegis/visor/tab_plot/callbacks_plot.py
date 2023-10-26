from dash import callback, Output, Input, ctx

from aegis.help.container import Container
from aegis.visor import funcs
from aegis.visor.tab_plot import make_plots
from aegis.visor.tab_plot import make_figures
from aegis.visor.tab_plot.static import FIGURE_INFO
from aegis.visor.tab_list.callbacks_list import SELECTION


containers = {}

funcs_dict = {
    "life expectancy": make_plots.get_life_expectancy,
    "intrinsic mortality": make_plots.get_intrinsic_mortality,
    "intrinsic survivorship": make_plots.get_intrinsic_survivorship,
    "fertility": make_plots.get_fertility,
    "cumulative reproduction": make_plots.get_cumulative_reproduction,
    "lifetime reproduction": make_plots.get_lifetime_reproduction,
    "birth structure": make_plots.get_birth_structure,
    "death structure": make_plots.get_death_structure,
    "total survivorship": make_plots.get_total_survivorship,
}


def make_figure(id_, plot_func):
    ys = [plot_func(containers[sim]) for sim in SELECTION]
    xs = make_figures.get_xs(id_, containers, ys)
    figure = make_figures.make_figure(id_, xs, ys)
    return figure


@callback(
    [Output(key, "figure") for key in FIGURE_INFO.keys()],
    Input("plot-view-button", "n_clicks"),
    Input("reload-plots-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def update_scatter_plot(*_):
    global containers

    triggered = ctx.triggered_id
    if triggered == "reload-plots-button":
        containers = {}

    for sim in SELECTION:
        if sim not in containers:
            containers[sim] = Container(funcs.BASE_DIR / sim)

    figures = {}

    # BUG no data saved yet on running simulations or interrupted simulations

    for id_, plot_func in funcs_dict.items():
        figures[id_] = make_figure(id_, plot_func)

    return [figures[key] for key in FIGURE_INFO.keys()]
