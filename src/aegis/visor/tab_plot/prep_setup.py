from dash import dcc
from aegis.visor.tab_plot import prep_x, prep_y

FIG_SETUP = {
    "bit states": {
        "title": "bit states",
        "prep_y": prep_y.get_bit_states,
        "prep_x": prep_x.get_stages,
        "prep_figure": "make_heatmap_figure",
        "description": dcc.Markdown(
            """
            Bit states for each site at a given stage. A population average. \n
            This plot can only display one selected simulation at a time.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "stage",
            "yaxis_title": "site",
        },
    },
    "causes of death": {
        "title": "causes of death",
        "prep_y": prep_y.get_causes_of_death,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_bar_figure",
        "description": dcc.Markdown(
            """...""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age",
            "yaxis_title": "",
        },
    },
    "derived allele frequencies": {
        "title": "derived allele frequencies",
        "prep_y": prep_y.get_derived_allele_freq,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_hist_figure",
        "description": dcc.Markdown(
            """...""",
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "derived allele frequency",
            "yaxis_title": "number of sites",
        },
    },
    "intrinsic mortality": {
        "title": "intrinsic mortality",
        "prep_y": prep_y.get_intrinsic_mortality,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_total_mortality,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_total_survivorship,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_intrinsic_survivorship,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_life_expectancy,
        "prep_x": prep_x.get_stages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_fertility,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_cumulative_reproduction,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_lifetime_reproduction,
        "prep_x": prep_x.get_stages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_birth_structure,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
        "prep_y": prep_y.get_death_structure,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_scatter_figure",
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
