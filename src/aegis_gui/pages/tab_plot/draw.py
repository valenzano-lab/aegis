import dash
from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot.plot.gen_fig import gen_fig
from aegis_gui.utilities import log_funcs


# Define a helper function to handle the logic
def handle_draw_plot(dropdown_values, selected_fig):
    if not dropdown_values or dropdown_values == [None]:
        return dash.no_update, dash.no_update, dash.no_update

    drag_maxs = []
    figures = []

    for fig_name in FIG_SETUP:
        if fig_name == selected_fig:  # Only update the figure that matches the selected tab
            figure, max_iloc = gen_fig(fig_name, dropdown_values, iloc=-1)
            figures.append(figure)
            drag_maxs.append(max_iloc)
        else:
            figures.append(dash.no_update)
            drag_maxs.append(dash.no_update)
    return figures, drag_maxs


# TODO take into consideration the existing slider value


@dash.callback(
    dash.Output({"type": "graph-figure", "index": dash.ALL}, "figure"),
    dash.Output({"type": "graph-slider", "index": dash.ALL}, "max"),
    dash.Output({"type": "graph-slider", "index": dash.ALL}, "value"),
    dash.Input("dropdown-multi", "value"),
    dash.Input("figure-select", "value"),
    dash.Input("refresh-figure-data", "n_clicks"),
    # prevent_initial_call=True,
)
@log_funcs.log_info
def draw_plot(dropdown_values, figure_selected, refresh_figure_data):
    figures, drag_maxs = handle_draw_plot(dropdown_values, figure_selected)
    return figures, drag_maxs, drag_maxs
