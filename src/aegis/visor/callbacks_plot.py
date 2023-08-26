import numpy as np
import plotly.graph_objs as go
from dash import callback, Output, Input

from aegis.help.container import Container
from aegis.visor import funcs
from aegis.visor.static import FIGURE_INFO

from aegis.visor.callbacks_results import SELECTION

# container = None
containers = {}

max_age = 50
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


@callback(
    [Output(key, "figure") for key in FIGURE_INFO.keys()],
    # Input("dynamic-dropdown", "value"),
    # Input("slider", "value"),
    Input("plot-view-button", "n_clicks"),
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
    for sim in SELECTION:
        if sim not in containers:
            containers[sim] = Container(f"/home/user/.local/share/aegis/{sim}")

    # marker_color_surv = "dodgerblue"
    # marker_color_repr = "crimson"

    fig_layout = dict(
        # yaxis={"range": [0, 1]},
        # xaxis={"range": [0, max_age]},
        width=300,
        height=300,
        margin={"t": 0, "r": 0, "b": 0, "l": 0},
        plot_bgcolor="rgba(0, 0, 0, 0.02)",
        # legend={"xanchor": "right", "yanchor": "bottom", "orientation": "v"},
        # paper_bgcolor="rgba(0, 0, 0, 0.1)",
    )

    def make_figure(xs, ys):
        return go.Figure(
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

    figures = {}

    # BUG no data saved yet on running simulations or interrupted simulations

    # Figure: life expectancy at age 0
    id_ = "life expectancy"

    def get_life_expectancy(sim):
        phenotypes = containers[sim].get_df("phenotypes")
        pdf = phenotypes.iloc[:, :max_age]
        survivorship = pdf.cumprod(1)
        y = survivorship.sum(1)
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_life_expectancy(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # Figure: intrinsic mortality
    id_ = "intrinsic mortality"

    def get_intrinsic_mortality(sim):
        phenotypes = containers[sim].get_df("phenotypes")
        pdf = phenotypes.iloc[-1, :max_age]
        y = 1 - pdf
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_intrinsic_mortality(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # Figure: intrinsic survivorship
    id_ = "intrinsic survivorship"

    def get_intrinsic_survivorship(sim):
        phenotypes = containers[sim].get_df("phenotypes")
        pdf = phenotypes.iloc[-1, :max_age]
        y = pdf.cumprod()
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_intrinsic_survivorship(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # Figure: fertility
    id_ = "fertility"

    def get_fertility(sim):
        phenotypes = containers[sim].get_df("phenotypes")
        fertility = phenotypes.iloc[-1, max_age:]
        y = fertility
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_fertility(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # Figure: cumulative reproduction
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13289
    # BUG fertility is 0 before maturity
    id_ = "cumulative reproduction"

    def get_cumulative_reproduction(sim):
        phenotypes = containers[sim].get_df("phenotypes")
        survivorship = phenotypes.iloc[-1, :max_age].cumprod()
        fertility = phenotypes.iloc[-1, max_age:]
        y = (survivorship.values * fertility.values).cumsum()
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_cumulative_reproduction(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # Figure: lifetime reproduction
    # BUG fertility is 0 before maturity
    id_ = "lifetime reproduction"

    def get_lifetime_reproduction(sim):
        phenotypes = containers[sim].get_df("phenotypes")
        survivorship = phenotypes.iloc[:, :max_age].cumprod(1)
        fertility = phenotypes.iloc[:, max_age:]
        y = np.sum((np.array(survivorship) * np.array(fertility)), axis=1)
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_lifetime_reproduction(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # Figure: birth structure
    id_ = "birth structure"

    def get_birth_structure(sim):
        age_at_birth = containers[sim].get_df("age_at_birth")
        y = age_at_birth.iloc[-1]
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_birth_structure(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # Figure: death structure
    id_ = "death structure"

    def get_death_structure(sim):
        age_at_genetic = containers[sim].get_df("age_at_genetic")
        age_at_overshoot = containers[sim].get_df("age_at_overshoot")

        t = -1
        pseudocount = 0
        y = (age_at_genetic.iloc[t] + pseudocount) / (
            age_at_overshoot.iloc[t] + age_at_genetic.iloc[t] + pseudocount
        )
        return y

    xs = [np.arange(1, max_age + 1) for sim in SELECTION]
    ys = [get_death_structure(sim) for sim in SELECTION]
    figures[id_] = make_figure(xs, ys)

    # aspect_ratio = 0.5

    for figure in figures.values():
        figure.update_layout(
            showlegend=False,
            # autosize=False,
            # width=800,  # Adjust the width as needed
            # height=int(
            # 800 * aspect_ratio
            # ),  # Calculate height to maintain the square aspect ratio
            # legend=dict(
            # x=1.02,  # Position of the legend along the x-axis (outside the plot)
            # y=0.5,  # Position of the legend along the y-axis (centered)
            # yanchor="middle",
            # xanchor="left",
            # ),
        )

    # for figure in figures.values():
    # figure.update_yaxes(showline=True)

    return [figures[key] for key in FIGURE_INFO.keys()]
