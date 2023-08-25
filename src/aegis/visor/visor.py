# Import packages
from dash import Dash, html, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go


from aegis.help.container import Container
from aegis.visor import funcs

from aegis.visor.static import FIGURE_INFO

import pathlib

funcs.hello()

# Figure descriptions start


# Figure description end


# Incorporate data
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)

container = None

paths_txt = pathlib.Path(__file__).absolute().parent.parent / "help" / "paths.txt"
with open(paths_txt, "r") as f:
    paths = f.read().split()[::-1]

# Initialize the app
app = Dash(__name__)
app._favicon = "favicon.ico"
app.title = "AEGIS visualizer"


# Create example plots
def create_plot(figure_name):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig = px.line(x=x, y=y, title=figure_name)
    return fig


# App layout
app.layout = html.Div(
    [
        # TITLE SECTION
        html.Div(
            className="title-section",
            children=[html.H1("AEGIS visor")],
        ),
        # CONTROL SECTION
        html.Div(
            className="control-section",
            children=[
                html.Button("Toggle Divs", id="toggle-button"),
            ],
        ),
        # SIMULATION SECTION
        html.Div(
            id="sim-section",
            children=[
                html.Button("run simulation", id="simulation-run-button"),
                dcc.Input(
                    id="config-make-text", type="text", placeholder="Enter text..."
                ),
                html.Button("make config", id="config-make-button"),
            ]
            + [
                html.Div(
                    children=[
                        html.Label(children=k),
                        dcc.Input(type="text", placeholder=str(v), id=f"config-{k}"),
                    ]
                )
                for k, v in funcs.DEFAULT_CONFIG_DICT.items()
                if not isinstance(v, list)
            ],
        ),
        # FIGURE SECTION
        html.Div(
            id="figure-section",
            style={"display": "none"},
            children=[
                html.Div(
                    className="dataset-section",
                    children=[
                        html.Button(
                            "reload list of simulations",
                            "load-paths-button",
                            style={"display": "none"},
                        ),
                        dcc.Dropdown(
                            id="dynamic-dropdown",
                            clearable=False,
                            style={"width": "50%"},
                        ),
                    ],
                ),
            ]
            + [
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id=figure_id,
                                    config={"displayModeBar": False},
                                    className="figure",
                                ),
                            ],
                            style={"padding-right": "20px"},
                        ),
                        html.Div(
                            children=[
                                html.P(
                                    children=info["title"], className="figure-title"
                                ),
                                html.P(
                                    children=info["description"],
                                    className="figure-description",
                                ),
                            ]
                        ),
                    ],
                    className="figure-card",
                )
                for figure_id, info in FIGURE_INFO.items()
            ]
            + [
                dcc.Graph(id="figurex"),
                html.Div(
                    [
                        dcc.Slider(
                            id="slider",
                            min=1,
                            max=10,
                            step=1,
                            value=5,
                            updatemode="drag",
                        ),
                    ],
                    # style={"width": "400px"},
                ),
            ],
        ),
        #
        # FOOTER SECTION
        html.Div([html.Hr(), html.P(children="aba baba ja sam zaba")]),
    ],
    className="main-container",
)


# load paths


@app.callback(
    Output("sim-section", "style"),
    Output("figure-section", "style"),
    Input("toggle-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_divs(n_clicks):
    div1_style = {"display": "block"}
    div2_style = {"display": "none"}

    if n_clicks % 2 == 1:
        div1_style, div2_style = div2_style, div1_style

    return div1_style, div2_style


@app.callback(
    Output("simulation-run-button", "style"), Input("simulation-run-button", "n_clicks")
)
def update_output(n_clicks):
    if n_clicks is not None:
        funcs.run()
    return {}


@app.callback(
    Output("config-make-button", "style"),
    Input("config-make-button", "n_clicks"),
    State("config-make-text", "value"),
    [
        State(f"config-{k}", "value")
        for k, v in funcs.DEFAULT_CONFIG_DICT.items()
        if not isinstance(v, list)
    ],
)
def update_output2(n_clicks, filename, *values):

    custom_config = {
        k: val
        for (k, v), val in zip(funcs.DEFAULT_CONFIG_DICT.items(), values)
        if not isinstance(v, list)
    }

    if n_clicks is not None:
        funcs.make_config_file(filename, custom_config)
    return {}


@callback(
    [Output("dynamic-dropdown", "options"), Output("dynamic-dropdown", "value")],
    [Input("load-paths-button", "n_clicks")],
)
def update_dropdown_options(n_clicks):

    with open(paths_txt, "r") as f:
        # decode paths.txt
        paths = f.read().split()[::-1]
        # remove duplicates with preserving order
        paths = sorted(set(paths), key=paths.index)
        # turn to pathlib.Path
        paths = [pathlib.Path(path) for path in paths]

    options = [
        {"label": f"{i}. {str(path.stem)} ({str(path)})", "value": str(path)}
        for i, path in enumerate(paths)
    ]

    return options, options[0]["value"]


max_age = 50
ages = np.arange(1, max_age + 1)


# def get_mortality_rate(phenotypes, max_age, t):
#     return 1 - phenotypes.iloc[t, :max_age]


# def get_survivorship(phenotypes, max_age, t):
#     return phenotypes.iloc[t, :max_age].cumprod()


# def get_fertility_rate(phenotypes, max_age, t):
#     return phenotypes.iloc[t, max_age:]


# def get_offspring_number(phenotypes, max_age, t):
#     return np.cumsum(
#         np.multiply(
#             list(phenotypes.iloc[t, max_age:]), list(phenotypes.iloc[t, :max_age])
#         )
#     )


# def get_birth_structure(age_at_birth, t):
#     return age_at_birth.iloc[t]


# def get_death_structure(age_at_genetic, age_at_overshoot, t):
#     pseudocount = 0
#     return (age_at_genetic.iloc[t] + pseudocount) / (
#         age_at_overshoot.iloc[t] + age_at_genetic.iloc[t] + pseudocount
#     )


@app.callback(
    [Output("slider", "max")],
    [Input("dynamic-dropdown", "value")],
)
def update_slider_config(selected_option):

    global container

    # initialize container if it is not
    if container is None:
        container = Container(selected_option)

    # update container if path changed
    if str(container.basepath) != str(selected_option):
        container = Container(selected_option)

    phenotypes = container.get_df("phenotypes")
    return (len(phenotypes),)


# Callback to handle dropdown option change
@app.callback(
    [Output(key, "figure") for key in FIGURE_INFO.keys()],
    [
        Input("dynamic-dropdown", "value"),
        Input("slider", "value"),
    ],
)
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


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
