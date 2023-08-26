from dash import dcc

FIGURE_INFO = {
    "life expectancy": {
        "title": "life expectancy at age 0",
        "description": dcc.Markdown(
            """
            Life expectancy at age 0
            $$e_0$$
            denotes the average expected lifespan at birth, i.e. how long does each individual live on average.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "stage",
            "yaxis_title": "",
        },
    },
    "intrinsic mortality": {
        "title": "intrinsic mortality",
        "description": """Probability to die """,
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "intrinsic survivorship": {
        "title": "intrinsic survivorship",
        "description": "asdjfkwejkre",
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "fertility": {
        "title": "fertility",
        "description": "Age-specific probability to produce a single offspring.",
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "cumulative reproduction": {
        "title": "cumulative reproduction",
        "description": "asdjfkwejkre",
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "lifetime reproduction": {
        "title": "lifetime reproduction",
        "description": "asdjfkwejkre",
        # graph
        "figure_layout": {
            "xaxis_title": "stage",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "birth structure": {
        "title": "birth structure",
        "description": "asdjfkwejkre",
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
    "death structure": {
        "title": "death structure",
        "description": "asdjfkwejkre",
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
            # "yaxis": {"range": [0, 1]},
        },
    },
}
