# Import packages
import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go


from aegis.help.container import Container

import pathlib

# Incorporate data
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)

container = None

paths_txt = pathlib.Path(__file__).absolute().parent / "paths.txt"
with open(paths_txt, "r") as f:
    paths = f.read().split()[::-1]

# Initialize the app
app = Dash(__name__)


# Create example plots
def create_plot(figure_name):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig = px.line(x=x, y=y, title=figure_name)
    return fig


# App layout
app.layout = html.Div(
    [
        # html.Link(
        #     rel="stylesheet", href="/styles.css"  # Path to your external CSS file
        # ),
        html.Button(
            "Load simulation data",
            "load-paths-button",
        ),
        dcc.Dropdown(
            id="dynamic-dropdown",
            clearable=False,
            style={"width": "50%"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="figure1",
                            config={"displayModeBar": False},
                            className="graph",
                        ),
                        dcc.Graph(
                            id="figure2",
                            config={"displayModeBar": False},
                            className="graph",
                        ),
                        dcc.Graph(
                            id="figure3",
                            config={"displayModeBar": False},
                            className="graph",
                        ),
                        dcc.Graph(
                            id="figure4",
                            config={"displayModeBar": False},
                            className="graph",
                        ),
                        dcc.Graph(
                            id="figure5",
                            config={"displayModeBar": False},
                            className="graph",
                        ),
                        dcc.Graph(
                            id="figure6",
                            config={"displayModeBar": False},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "row"},
                ),
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
                    style={"width": "400px"},
                ),
            ]
        ),
    ]
)


# load paths


@callback(
    [Output("dynamic-dropdown", "options"), Output("dynamic-dropdown", "value")],
    [Input("load-paths-button", "n_clicks")],
)
def update_dropdown_options(n_clicks):

    with open(paths_txt, "r") as f:
        paths = f.read().split()[::-1]

    options = [{"label": path, "value": path} for path in paths]

    return options, options[0]["value"]


max_age = 50
ages = np.arange(1, max_age + 1)


def get_mortality_rate(phenotypes, max_age, t):
    return 1 - phenotypes.iloc[t, :max_age]


def get_survivorship(phenotypes, max_age, t):
    return phenotypes.iloc[t, :max_age].cumprod()


def get_fertility_rate(phenotypes, max_age, t):
    return phenotypes.iloc[t, max_age:]


def get_offspring_number(phenotypes, max_age, t):
    return np.cumsum(
        np.multiply(
            list(phenotypes.iloc[t, max_age:]), list(phenotypes.iloc[t, :max_age])
        )
    )


def get_birth_structure(age_at_birth, t):
    return age_at_birth.iloc[t]


def get_death_structure(age_at_genetic, age_at_overshoot, t):
    pseudocount = 0
    return (age_at_genetic.iloc[t] + pseudocount) / (
        age_at_overshoot.iloc[t] + age_at_genetic.iloc[t] + pseudocount
    )


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
    [
        Output("figure1", "figure"),
        Output("figure2", "figure"),
        Output("figure3", "figure"),
        Output("figure4", "figure"),
        Output("figure5", "figure"),
        Output("figure6", "figure"),
    ],
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
    genotypes = container.get_df("genotypes")

    fig_layout = dict(
        yaxis={"range": [0, 1]}, xaxis={"range": [0, max_age]}, width=400, height=400
    )
    fig_data = {
        "mode": "markers",
        "x": ages,
    }

    figure1 = go.Figure(
        data=[
            go.Scatter(
                y=get_mortality_rate(phenotypes, max_age, slider_input - 1),
                name="Scatter Plot",
                **fig_data,
            )
        ],
        layout=go.Layout(
            title=slider_input,
            xaxis_title="age",
            yaxis_title="mortality rate",
            **fig_layout,
        ),
    )

    figure2 = go.Figure(
        data=[
            go.Scatter(
                y=get_survivorship(phenotypes, max_age, slider_input - 1),
                name="Scatter Plot",
                **fig_data,
            )
        ],
        layout=go.Layout(
            title=slider_input,
            xaxis_title="age",
            yaxis_title="survivorship",
            **fig_layout,
        ),
    )

    figure3 = go.Figure(
        data=[
            go.Scatter(
                y=get_fertility_rate(phenotypes, max_age, slider_input - 1),
                name="Scatter Plot",
                **fig_data,
            )
        ],
        layout=go.Layout(
            title=slider_input,
            xaxis_title="age",
            yaxis_title="fertility rate",
            **fig_layout,
        ),
    )

    figure4 = go.Figure(
        data=[
            go.Scatter(
                y=get_offspring_number(phenotypes, max_age, slider_input - 1),
                name="Scatter Plot",
                **fig_data,
            )
        ],
        layout=go.Layout(
            title=slider_input,
            xaxis_title="age",
            yaxis_title="age-cumulative # of offspring per individual",
            **fig_layout,
        ),
    )

    figure5 = go.Figure(
        data=[
            go.Scatter(
                y=get_birth_structure(age_at_birth, slider_input - 1),
                name="Scatter Plot",
                **fig_data,
            )
        ],
        layout=go.Layout(
            title=slider_input,
            xaxis_title="age",
            yaxis_title="# of offspring born to parents of age x",
            **fig_layout,
        ),
    )
    figure6 = go.Figure(
        data=[
            go.Scatter(
                y=get_death_structure(
                    age_at_genetic, age_at_overshoot, slider_input - 1
                ),
                name="Scatter Plot",
                **fig_data,
            )
        ],
        layout=go.Layout(
            title=slider_input,
            xaxis_title="age",
            yaxis_title="proportion of death that is intrinsic",
            **fig_layout,
        ),
    )

    return figure1, figure2, figure3, figure4, figure5, figure6


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
