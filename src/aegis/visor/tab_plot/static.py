from dash import dcc
from aegis.visor.tab_plot import make_plots


FIGURE_INFO = {
    "intrinsic mortality": {
        "title": "intrinsic mortality",
        "plotter": make_plots.get_intrinsic_mortality,
        "description": dcc.Markdown(
            """Genetic (individual-specific, heritable) mortality at a given age. A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "total mortality": {
        "title": "total mortality",
        "plotter": make_plots.get_total_mortality,
        "description": dcc.Markdown(
            """Observed (individual-specific, heritable) mortality at a given age (all sources of mortality considered).
            A population average.
            Some data points are missing because mortality cannot be calculated for ages at which there are no individuals alive.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "total survivorship": {
        "title": "total survivorship",
        "plotter": make_plots.get_total_survivorship,
        "description": dcc.Markdown(
            """Observed expected probability to survive to a specific age (when all sources of mortality are considered). A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "intrinsic survivorship": {
        "title": "intrinsic survivorship",
        "plotter": make_plots.get_intrinsic_survivorship,
        "description": dcc.Markdown(
            """Inferred expected probability to survive to a specific age only given genetic mortality. A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "life expectancy": {
        "title": "life expectancy at age 0",
        "plotter": make_plots.get_life_expectancy,
        "description": dcc.Markdown(
            """
            Expected lifespan at birth over the course of the simulation. A population average.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "stage",
            "yaxis_title": "",
        },
    },
    "fertility": {
        "title": "fertility",
        "plotter": make_plots.get_fertility,
        "description": dcc.Markdown(
            """The probability to produce a single offspring at each age. A population average.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "cumulative reproduction": {
        "title": "cumulative reproduction",
        "plotter": make_plots.get_cumulative_reproduction,
        "description": dcc.Markdown(
            """The expected number of produced offspring until a given age. A population average.
            """,
            # \n$\sum_{x=0}^{a^*}m(x)l(x)$
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "lifetime reproduction": {
        "title": "lifetime reproduction",
        "plotter": make_plots.get_lifetime_reproduction,
        "description": dcc.Markdown(
            """The expected number of offspring produced until death over the course of the simulation. A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "stage",
            "yaxis_title": "",
        },
    },
    "birth structure": {
        "title": "birth structure",
        "plotter": make_plots.get_birth_structure,
        "description": dcc.Markdown(
            """The proportion of newborns produced by parents of a given age.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "death structure": {
        "title": "death structure",
        "plotter": make_plots.get_death_structure,
        "description": dcc.Markdown(
            """The measured ratio of intrinsic deaths versus total deaths, grouped by age.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    # "total survivorship": {
    #     "title": "total survivorship",
    #     "description": dcc.Markdown(
    #         """xxx.""",
    #         mathjax=True,
    #     ),
    #     # graph
    #     "figure_layout": {
    #         "xaxis_title": "age",
    #         "yaxis_title": "",
    #         # "yaxis": {"range": [0, 1]},
    #     },
    # },
}
