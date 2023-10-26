import numpy as np
import plotly.graph_objs as go
from dash import callback, Output, Input, ctx

from aegis.help.container import Container
from aegis.visor import funcs
from aegis.visor.static import FIGURE_INFO
from aegis.visor.plot import plotter

from aegis.visor.callbacks_results import SELECTION

# container = None
containers = {}
# ages = np.arange(1, max_age + 1)

# colors = ["dodgerblue", "hotpink", "crimson"]


# @callback(
#     [
#         Output("dynamic-dropdown", "options"),
#         Output("dynamic-dropdown", "value"),
#     ],
#     [
#         Input("plot-view-button", "n_clicks"),
#     ],
#     prevent_initial_call=True,
# )
# @funcs.print_function_name
# def refresh_dropdown_options(*_):
#     paths = funcs.get_sim_paths()
#     dropdown_options = [
#         {"label": f"{i}. {str(path.stem)} ({str(path)})", "value": str(path)}
#         for i, path in enumerate(paths)
#     ]
#     # BUG fix if no dropdown_options available
#     return dropdown_options, dropdown_options[0]["value"]


# @callback(
#     [Output("slider", "max")],
#     [Input("dynamic-dropdown", "value")],
#     prevent_initial_call=True,
# )
# @funcs.print_function_name
# def update_slider(selected_option):

#     global container

#     # initialize container if it is not
#     if container is None:
#         container = Container(selected_option)

#     # update container if path changed
#     if str(container.basepath) != str(selected_option):
#         container = Container(selected_option)

#     phenotypes = container.get_df("phenotypes")
#     return (len(phenotypes),)


# @callback(
#     Output("reload-plots-button", "style"),
#     Input("reload-plots-button", "n_clicks"),
#     prevent_initial_call=True,
# )
# def reload_plots(n_clicks):
#     global containers
#     containers = {}


@callback(
    [Output(key, "figure") for key in FIGURE_INFO.keys()],
    # Input("dynamic-dropdown", "value"),
    # Input("slider", "value"),
    Input("plot-view-button", "n_clicks"),
    Input("reload-plots-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def update_scatter_plot(*_):
    # global container

    # # initialize container if it is not
    # if container is None:
    #     container = Container(selected_option)

    # # update container if path changed
    # if str(container.basepath) != str(selected_option):
    #     container = Container(selected_option)

    # phenotypes = container.get_df("phenotypes")
    # age_at_birth = container.get_df("age_at_birth")
    # age_at_genetic = container.get_df("age_at_genetic")
    # age_at_overshoot = container.get_df("age_at_overshoot")
    # # genotypes = container.get_df("genotypes")

    global containers

    triggered = ctx.triggered_id
    if triggered == "reload-plots-button":
        containers = {}

    for sim in SELECTION:
        if sim not in containers:
            containers[sim] = Container(funcs.BASE_DIR / sim)

    # marker_color_surv = "dodgerblue"
    # marker_color_repr = "crimson"

    fig_layout = dict(
        # yaxis={"range": [0, 1]},
        width=300,
        height=300,
        margin={"t": 0, "r": 0, "b": 0, "l": 0},
        plot_bgcolor="rgba(0, 0, 0, 0.02)",
        # legend={"xanchor": "right", "yanchor": "bottom", "orientation": "v"},
        paper_bgcolor="rgba(24, 25, 27, 0)",
        xaxis=dict(
            # range=[0, 1],
            gridcolor="rgb(46, 49, 51)",
            linecolor="rgb(46, 49, 51)",
            tickfont=dict(
                color="rgb(190, 189, 183)",
            ),  # Change the x-axis tick label color
            titlefont=dict(
                color="rgb(190, 189, 183)",
            ),  # Change the y-axis tick label color
        ),
        yaxis=dict(
            # range=[0, 1],
            gridcolor="rgb(46, 49, 51)",
            linecolor="rgb(46, 49, 51)",
            tickfont=dict(
                color="rgb(190, 189, 183)",
            ),  # Change the y-axis tick label color
            titlefont=dict(
                color="rgb(190, 189, 183)",
            ),  # Change the y-axis tick label color
        ),
    )

    def make_figure(xs, ys):
        figure = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=sim,
                )
                for x, y, sim in zip(xs, ys, SELECTION)
            ],
            layout=go.Layout(
                **FIGURE_INFO[id_]["figure_layout"],
                **fig_layout,
            ),
        )

        maxx = max(max(x) for x in xs) if ys else 1
        maxy = max(max(y) for y in ys) if ys else 1

        figure.update_layout(
            xaxis_range=[0, 1 if maxx < 1 else maxx * 1.05],
            yaxis_range=[0, 1 if maxy < 1 else maxy * 1.05],
            showlegend=False,
        )

        return figure

    figures = {}

    # BUG no data saved yet on running simulations or interrupted simulations

    def get_xs(id_, ys):
        type_ = FIGURE_INFO[id_]["figure_layout"]["xaxis_title"]
        if type_ == "stage":
            return [
                np.arange(1, len(y) + 1) * containers[sim].get_config()["VISOR_RATE_"]
                for sim, y in zip(SELECTION, ys)
            ]
        else:
            return [
                np.arange(1, containers[sim].get_config()["MAX_LIFESPAN"] + 1)
                for sim in SELECTION
            ]

    id_ = "life expectancy"
    ys = [plotter.get_life_expectancy(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    id_ = "intrinsic mortality"
    ys = [plotter.get_intrinsic_mortality(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    id_ = "intrinsic survivorship"
    ys = [plotter.get_intrinsic_survivorship(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    id_ = "fertility"
    ys = [plotter.get_fertility(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    id_ = "cumulative reproduction"
    ys = [plotter.get_cumulative_reproduction(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    id_ = "lifetime reproduction"
    ys = [plotter.get_lifetime_reproduction(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    id_ = "birth structure"
    ys = [plotter.get_birth_structure(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    id_ = "death structure"
    ys = [plotter.get_death_structure(containers[sim]) for sim in SELECTION]
    xs = get_xs(id_, ys)
    figures[id_] = make_figure(xs, ys)

    return [figures[key] for key in FIGURE_INFO.keys()]
