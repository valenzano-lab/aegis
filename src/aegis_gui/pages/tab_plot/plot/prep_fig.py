from aegis_gui.pages.tab_plot.plot.prep_setup import FIG_SETUP
import plotly.graph_objs as go

bootstrap_colors = [
    "#007bff",  # primary
    "#6c757d",  # secondary
    "#28a745",  # success
    "#dc3545",  # danger
    "#ffc107",  # warning
    "#17a2b8",  # info
    "#f8f9fa",  # light
    "#343a40",  # dark
]

FIG_xaxis = dict(
    # showgrid=False,
    zeroline=False,
    # showline=False,
    nticks=10,
    # gridcolor="#cccccc",
    griddash="dot",
    # gridwidth=1,
    linecolor="rgb(46, 49, 51)",
)

FIG_yaxis = dict(
    # showgrid=False,
    zeroline=False,
    # showline=False,
    nticks=10,
    # gridcolor="#cccccc",
    griddash="dot",
    # gridwidth=1,
    linecolor="rgb(46, 49, 51)",
)

FIG_legend = dict(x=0.97, y=1, yanchor="top", xanchor="right", orientation="v")

FIG_LAYOUT = dict(
    width=300 * 1.1,
    height=300 * 1.1,
    # margin={"t": 0, "r": 0, "b": 0, "l": 0},
    margin={"t": 0, "r": 0, "b": 0, "l": 0},
    plot_bgcolor="rgba(150, 150, 200, 0.1)",
    paper_bgcolor="rgba(24, 25, 27, 0.0)",
    # font_color="white",
    # showlegend=True,
    # legend=dict(x=1.05, y=1),
    # showlegend=False,
    showlegend=True,
    # legend=dict(x=0, y=0),
    legend=FIG_legend,
    font_size=14,
    # font_family="Inter",
    xaxis=FIG_xaxis,
    yaxis=FIG_yaxis,
)

dark_mode_white = "rgba(255,255,255,0.8)"


FIG_LAYOUT_DARK_MODE = dict(
    plot_bgcolor="rgba(0, 0, 0, 0.1)",  # Darker background for the plot area
    paper_bgcolor="rgba(24, 25, 27, 0.0)",  # Darker background for the overall figure
    font_color=dark_mode_white,  # Light font color for better contrast
    legend={**FIG_legend, **dict(font=dict(color=dark_mode_white))},  # Light legend text
    xaxis={
        **FIG_xaxis,
        **dict(
            linecolor=dark_mode_white,  # Light color for x-axis line
            tickfont=dict(color=dark_mode_white),  # Light color for x-axis ticks
            gridcolor="rgba(255, 255, 255, 0.1)",  # Light grid lines for visibility
        ),
    },
    yaxis={
        **FIG_yaxis,
        **dict(
            linecolor=dark_mode_white,  # Light color for y-axis line
            tickfont=dict(color=dark_mode_white),  # Light color for y-axis ticks
            gridcolor="rgba(255, 255, 255, 0.1)",  # Light grid lines for visibility
        ),
    },
)


def make_empty_figure(dark_mode=False):
    return go.Figure(layout=go.Layout({**FIG_LAYOUT, **(FIG_LAYOUT_DARK_MODE if dark_mode else {})}))


def make_scatter_figure(id_, xs, ys, selected_sims, dark_mode=False):
    figure = go.Figure(
        data=[
            go.Scatter(
                x=x, y=y, mode="markers", name=sim, marker=dict(color=bootstrap_colors[i % len(bootstrap_colors)])
            )
            for i, (x, y, sim) in enumerate(zip(xs, ys, selected_sims))
        ],
        layout=go.Layout(
            {**FIG_LAYOUT, **(FIG_LAYOUT_DARK_MODE if dark_mode else {}), **FIG_SETUP[id_]["figure_layout"]}
        ),
    )

    # Compute maximum axes limits
    maxx = max(max(x) for x in xs) if ys else 1
    maxy = max(max(y) for y in ys) if ys else 1

    # Update axes
    figure.update_xaxes(
        range=[0, 1 if maxx < 1 else maxx * 1.05],
    )
    figure.update_yaxes(
        range=[0, 1.05 if maxy < 1 else maxy * 1.1],
    )

    # Custom plots
    if id_ == "birth table":
        figure.update_yaxes(range=[0, maxy * 1.05])

    return figure


def make_line_figure(id_, xs, ys, selected_sims, dark_mode=False):
    figure = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="lines", name=sim, line=dict(color=bootstrap_colors[i % len(bootstrap_colors)]))
            for i, (x, y, sim) in enumerate(zip(xs, ys, selected_sims))
        ],
        layout=go.Layout(
            {**FIG_LAYOUT, **(FIG_LAYOUT_DARK_MODE if dark_mode else {}), **FIG_SETUP[id_]["figure_layout"]}
        ),
    )

    # Compute maximum axes limits
    maxx = max(max(x) for x in xs) if xs else 1
    maxy = max(max(y) for y in ys) if ys else 1

    # Update axes
    figure.update_xaxes(
        range=[0, 1 if maxx < 1 else maxx * 1.05],
    )
    figure.update_yaxes(
        range=[0, 1.05 if maxy < 1 else maxy * 1.1],
    )

    # Custom plots
    if id_ == "birth table":
        figure.update_yaxes(range=[0, maxy * 1.05])

    return figure


def make_hist_figure(id_, xs, ys, selected_sims, dark_mode=False):
    fig_setup = FIG_SETUP[id_]
    figure = go.Figure(
        data=[
            go.Histogram(
                x=y,
                name=sim,
                nbinsx=fig_setup["nbinsx"] if "nbinsx" in fig_setup else None,
                marker_color=bootstrap_colors[i % len(bootstrap_colors)],
            )
            for i, (y, sim) in enumerate(zip(ys, selected_sims))
        ],
        layout=go.Layout(
            {**FIG_LAYOUT, **(FIG_LAYOUT_DARK_MODE if dark_mode else {}), **FIG_SETUP[id_]["figure_layout"]}
        ),
    )

    return figure


def make_heatmap_figure(id_, xs, ys, selected_sims, dark_mode=False):
    x = xs[0]
    y = ys[0]
    sim = next(iter(selected_sims))
    figure = go.Figure(
        data=go.Heatmap(z=y.T, x=x, name=sim, colorscale="Electric", showscale=True),
        layout=go.Layout(
            {**FIG_LAYOUT, **(FIG_LAYOUT_DARK_MODE if dark_mode else {}), **FIG_SETUP[id_]["figure_layout"]}
        ),
    )

    figure.update_yaxes(
        autorange="reversed",
        # ticks="",
        # showticklabels=False,
    )

    return figure


def make_bar_figure_stacked(id_, xs, ys, selected_sims, dark_mode=False):
    assert len(ys) > 0, f"{id_}"
    y = ys[0]
    figure = go.Figure(
        data=[
            go.Bar(x=y.index, y=y.loc[:, i], width=0.7, name=i, marker=dict(line=dict(width=0 if dark_mode else 0)))
            for i in y.columns
        ],
        layout=go.Layout(
            {**FIG_LAYOUT, **(FIG_LAYOUT_DARK_MODE if dark_mode else {}), **FIG_SETUP[id_]["figure_layout"]},
            barmode="stack",
            showlegend=True,
            # width=420,
        ),
    )
    return figure


def make_bar_figure_not_stacked(id_, xs, ys, selected_sims, dark_mode=False):

    figure = go.Figure(
        data=[
            go.Bar(x=x, y=y, name=sim, marker=dict(line=dict(width=0 if dark_mode else 0)))
            # for i in columns
            for i, (x, y, sim) in enumerate(zip(xs, ys, selected_sims))
        ],
        layout=go.Layout(
            {
                **FIG_LAYOUT,
                **(FIG_LAYOUT_DARK_MODE if dark_mode else {}),
                **FIG_SETUP[id_]["figure_layout"],
            },
            barmode="group",
            showlegend=True,
            bargroupgap=0.1,
            bargap=0.15,
            # width=420,
        ),
    )
    return figure
