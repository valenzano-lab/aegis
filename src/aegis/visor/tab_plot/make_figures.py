from aegis.visor.tab_plot.static import FIGURE_INFO

from aegis.visor.tab_list.callbacks_list import SELECTION
import numpy as np
import plotly.graph_objs as go

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


def get_xs(id_, containers, ys):
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


def make_figure(id_, xs, ys):
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
