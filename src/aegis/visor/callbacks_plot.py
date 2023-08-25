import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, html, dcc, callback, Output, Input, State

from aegis.help.container import Container
from aegis.visor import funcs
from aegis.visor.static import FIGURE_INFO

container = None
max_age = 50
ages = np.arange(1, max_age + 1)


@callback(
    [Output("slider", "max")],
    [Input("dynamic-dropdown", "value")],
    prevent_initial_call=True,
)
@funcs.print_function_name
def update_slider(selected_option):

    global container

    # initialize container if it is not
    if container is None:
        container = Container(selected_option)

    # update container if path changed
    if str(container.basepath) != str(selected_option):
        container = Container(selected_option)

    phenotypes = container.get_df("phenotypes")
    return (len(phenotypes),)


@callback(
    [Output(key, "figure") for key in FIGURE_INFO.keys()],
    Input("dynamic-dropdown", "value"),
    Input("slider", "value"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def update_scatter_plot(selected_option, slider_input):

    global container

    # initialize container if it is not
    if container is None:
        container = Container(selected_option)

    # update container if path changed
    if str(container.basepath) != str(selected_option):
        container = Container(selected_option)

    phenotypes = container.get_df("phenotypes")
    age_at_birth = container.get_df("age_at_birth")
    age_at_genetic = container.get_df("age_at_genetic")
    age_at_overshoot = container.get_df("age_at_overshoot")
    # genotypes = container.get_df("genotypes")

    marker_color_surv = "dodgerblue"
    marker_color_repr = "crimson"

    fig_layout = dict(
        # yaxis={"range": [0, 1]},
        # xaxis={"range": [0, max_age]},
        width=300,
        height=300,
        margin={"t": 0, "r": 0, "b": 0, "l": 0},
        plot_bgcolor="rgba(0, 0, 0, 0.02)",
        # paper_bgcolor="rgba(0, 0, 0, 0.1)",
    )

    figures = {}

    # Figure: life expectancy at age 0
    id_ = "life expectancy"
    pdf = phenotypes.iloc[:, :max_age]
    survivorship = pdf.cumprod(1)
    y = survivorship.sum(1)
    x = np.arange(len(y))

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_surv})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # Figure: intrinsic mortality
    id_ = "intrinsic mortality"
    pdf = phenotypes.iloc[-1, :max_age]
    y = 1 - pdf
    x = np.arange(max_age) + 1

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_surv})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # Figure: intrinsic survivorship
    id_ = "intrinsic survivorship"
    pdf = phenotypes.iloc[-1, :max_age]
    y = pdf.cumprod()
    x = np.arange(max_age) + 1

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_surv})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # Figure: fertility
    id_ = "fertility"
    fertility = phenotypes.iloc[-1, max_age:]
    y = fertility
    x = np.arange(max_age) + 1

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_repr})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # Figure: cumulative reproduction
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13289
    # BUG fertility is 0 before maturity
    id_ = "cumulative reproduction"
    survivorship = phenotypes.iloc[-1, :max_age].cumprod()
    fertility = phenotypes.iloc[-1, max_age:]
    y = (survivorship.values * fertility.values).cumsum()
    x = np.arange(max_age) + 1

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_repr})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # Figure: lifetime reproduction
    # BUG fertility is 0 before maturity
    id_ = "lifetime reproduction"
    survivorship = phenotypes.iloc[:, :max_age].cumprod(1)
    fertility = phenotypes.iloc[:, max_age:]
    y = np.sum((np.array(survivorship) * np.array(fertility)), axis=1)
    x = np.arange(len(y))

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_repr})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # Figure: birth structure
    id_ = "birth structure"

    y = age_at_birth.iloc[-1]
    x = np.arange(len(y))

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_repr})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # Figure: death structure
    id_ = "death structure"

    t = -1
    pseudocount = 0
    y = (age_at_genetic.iloc[t] + pseudocount) / (
        age_at_overshoot.iloc[t] + age_at_genetic.iloc[t] + pseudocount
    )
    x = np.arange(len(y))[y.notna()]

    figures[id_] = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", marker={"color": marker_color_repr})
        ],
        layout=go.Layout(
            **FIGURE_INFO[id_]["figure_layout"],
            **fig_layout,
        ),
    )

    # for figure in figures.values():
    # figure.update_yaxes(showline=True)

    return [figures[key] for key in FIGURE_INFO.keys()]
