from dash import dcc

FIGURE_INFO = {
    "intrinsic mortality": {
        "title": "intrinsic mortality",
        "description": dcc.Markdown(
            """Genetic (individual-specific, heritable) mortality at a given age. A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "total survivorship": {
        "title": "total survivorship",
        "description": dcc.Markdown(
            """Observed expected probability to survive to a specific age (when all sources of mortality are considered). A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "intrinsic survivorship": {
        "title": "intrinsic survivorship",
        "description": dcc.Markdown(
            """Inferred expected probability to survive to a specific age only given genetic mortality. A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "life expectancy": {
        "title": "life expectancy at age 0",
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
        "description": dcc.Markdown(
            """The probability to produce a single offspring at each age. A population average.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "cumulative reproduction": {
        "title": "cumulative reproduction",
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
            # "yaxis": {"range": [0, 1]},
        },
    },
    "lifetime reproduction": {
        "title": "lifetime reproduction",
        "description": dcc.Markdown(
            """The expected number of offspring produced until death over the course of the simulation. A population average.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "stage",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "birth structure": {
        "title": "birth structure",
        "description": dcc.Markdown(
            """The proportion of newborns produced by parents of a given age.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "death structure": {
        "title": "death structure",
        "description": dcc.Markdown(
            """The measured ratio of intrinsic deaths versus total deaths, grouped by age.""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
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
