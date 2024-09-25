import dash
from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
from aegis_gui.pages.tab_plot.plot.gen_fig import generate_figure
from aegis_gui.pages.tab_plot.plot.prep_fig import make_empty_figure
from aegis_gui.utilities import log


# Define a helper function to handle the logic
def handle_draw_plot(dropdown_values, selected_fig, dark_mode):
    if not dropdown_values or dropdown_values == [None]:
        return dash.no_update, dash.no_update

    drag_maxs = []
    figures = []

    for fig_name in FIG_SETUP:
        if fig_name == selected_fig:  # Only update the figure that matches the selected tab
            figure, max_iloc = generate_figure(fig_name, dropdown_values, iloc=-1, dark_mode=dark_mode)
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
    dash.State("color-mode-switch", "value"),
    # prevent_initial_call=True,
)
@log.log_debug
def draw_plot(dropdown_values, figure_selected, refresh_figure_data, dark_mode):
    # Return empty figure and no updates for sliders if dropdown_values is None
    if dropdown_values == []:
        # BUG it complains
        n = len(FIG_SETUP)
        empty_figures = [make_empty_figure()] * n  # Return an empty figure for each graph
        return empty_figures, [0] * n, [1] * n

    # Otherwise, handle the plot drawing logic
    figures, drag_maxs = handle_draw_plot(dropdown_values, figure_selected, dark_mode=dark_mode)
    return figures, drag_maxs, drag_maxs
