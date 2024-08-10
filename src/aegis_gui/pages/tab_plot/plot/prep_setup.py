import dash
import dash_bootstrap_components as dbc
from aegis_gui.pages.tab_plot.plot import prep_y
from aegis_gui.pages.tab_plot.plot import prep_x


FIG_SETUP = {
    "bit states": {
        "title": "bit states",
        "supports_multi": False,
        "prep_y": prep_y.get_bit_states,
        "prep_x": prep_x.get_steps_multiplied,
        "prep_figure": "make_heatmap_figure",
        "description": dash.dcc.Markdown(
            """
            Bit states for each site at a given step.
            \n
            Population averages. Interval averages.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "genome site",
        },
    },
    "death table": {
        "title": "death table",
        "supports_multi": False,
        "prep_y": prep_y.get_death_table,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_bar_figure",
        "description": dash.dcc.Markdown(
            """
            Number of deaths per age class, stratified by cause.
            \n
            Interval averages.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "number of deaths",
        },
    },
    "derived allele frequencies": {
        "title": "derived allele frequencies",
        "supports_multi": False,
        "prep_y": prep_y.get_derived_allele_freq,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_hist_figure",
        "description": dash.dcc.Markdown(
            """
            Sites with derived allele frequency of 0 are ignored.
            Ancestral states were the most common states an interval ago.
            \n
            """,
            mathjax=True,
        ),
        "nbinsx": 10,
        # graph
        "figure_layout": {
            "xaxis_title": "derived allele frequency",
            "yaxis_title": "number of genome sites",
            "xaxis": {
                "range": [0, 1],
            },
            # Change log scale of y axis
            # "yaxis": {
            #     "type": "log",
            # },
        },
    },
    "intrinsic mortality": {
        "title": "intrinsic mortality",
        "supports_multi": True,
        "prep_y": prep_y.get_mortality_intrinsic,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.html.Div(
            [
                dash.dcc.Markdown(
                    """
            Genetic (individual-specific, heritable) mortality at a given age.
            \n
            Population medians.
            """,
                    mathjax=True,
                ),
                # dash.html.Span(
                #     "tooltips", id="tooltip-target-1", style={"textDecoration": "underline", "cursor": "pointer"}
                # ),
                # dbc.Tooltip("This is the first tooltip", target="tooltip-target-1"),
            ]
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "intrinsic mortality",
        },
    },
    "total mortality": {
        "title": "total mortality",
        "supports_multi": True,
        "prep_y": prep_y.get_mortality_observed,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            Observed (individual-specific, heritable) mortality at a given age (all sources of mortality considered).
            \n
            Population averages. Interval averages.
            \n
            Missing data points are for age classes at which no living individuals have been observed.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "total mortality",
        },
    },
    # BUG goes up and down when the snapshot_rate is very low
    "total survivorship": {
        "title": "total survivorship",
        "supports_multi": True,
        "prep_y": prep_y.get_total_survivorship,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            Observed expected probability to survive to a specific age (when all sources of mortality are considered).
            \n
            Population averages. Interval averages.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "total survivorship",
        },
    },
    "intrinsic survivorship": {
        "title": "intrinsic survivorship",
        "supports_multi": True,
        "prep_y": prep_y.get_intrinsic_survivorship,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            # TODO clarify
            """
            Expected probability to survive to a specific age class only given genetic mortality.
            \n
            Computed using the median intrinsic mortalities.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "intrinsic survivorship",
        },
    },
    "life expectancy": {
        "title": "life expectancy at age 0",
        "supports_multi": True,
        "prep_y": prep_y.get_life_expectancy,
        "prep_x": prep_x.get_steps_multiplied,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            # TODO check this
            """
            Expected lifespan at birth. Plotted over the course of the simulation.
            \n
            Population averages.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "life expectancy",
        },
    },
    "population size": {
        "title": "population size",
        "supports_multi": True,
        "prep_y": prep_y.get_population_size,
        "prep_x": prep_x.get_steps_non_multiplied,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            # TODO check this
            """
            .
            \n
            .
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "life expectancy",
        },
    },
    "fertility": {
        "title": "intrinsic fertility",
        "supports_multi": True,
        "prep_y": prep_y.get_fertility,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            The intrinsic probability to produce a single offspring at each age class.
            \n
            Population medians.
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "intrinsic fertility",
        },
    },
    # "cumulative reproduction": {
    #     "title": "cumulative reproduction",
    # "supports_multi": True,
    # "prep_y": prep_y.get_cumulative_reproduction,
    #     "prep_x": prep_x.get_ages,
    #     "prep_figure": "make_line_figure",
    #     "description": dash.dcc.Markdown(
    #         """
    #         The expected number of offspring produced per individual until a given age class.
    #         \n
    #         Population averages.
    #         """,
    #         # \n$\sum_{x=0}^{a^*}m(x)l(x)$
    #         mathjax=True,
    #     ),
    #     # graph
    #     "figure_layout": {
    #         "xaxis_title": "age class",
    #         "yaxis_title": "cumulative number of offspring",
    #     },
    # },
    # "lifetime reproduction": {
    #     "title": "lifetime reproduction",
    # "supports_multi": True,
    # "prep_y": prep_y.get_lifetime_reproduction,
    #     "prep_x": prep_x.get_steps_multiplied,
    #     "prep_figure": "make_line_figure",
    #     "description": dash.dcc.Markdown(
    #         """
    #         The expected number of offspring produced per individual until death. Plotted over the course of the simulation.
    #         \n
    #         Population averages.
    #         """,
    #         mathjax=True,
    #     ),
    #     # graph
    #     "figure_layout": {
    #         "xaxis_title": "simulation step",
    #         "yaxis_title": "lifetime number of offspring",
    #     },
    # },
    "birth table": {
        "title": "birth table",
        "supports_multi": True,
        "prep_y": prep_y.get_birth_table,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            The number of newborns produced by parents of a given age class.
            \n
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "number of newborns",
        },
    },
    "life table": {
        "title": "life table",
        "supports_multi": True,
        "prep_y": prep_y.get_life_table,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            The proportion of living individuals.
            \n
            """,
            mathjax=True,
        ),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "proportion of living individuals",
        },
    },
    # "death structure": {
    #     "title": "death structure",
    # "supports_multi": True,
    # "prep_y": prep_y.get_death_structure,
    #     "prep_x": prep_x.get_ages,
    #     "prep_figure": "make_line_figure",
    #     "description": dash.dcc.Markdown(
    #         """The measured ratio of intrinsic deaths versus total deaths, grouped by age.""",
    #         mathjax=True,
    #     ),
    #     # graph
    #     "figure_layout": {
    #         "xaxis_title": "age class",
    #         "yaxis_title": "",
    #     },
    # },
    # "total survivorship": {
    #     "title": "total survivorship",
    # "supports_multi": True,
    # "description": dash.dcc.Markdown(
    #         """xxx.""",
    #         mathjax=True,
    #     ),
    #     # graph
    #     "figure_layout": {
    #         "xaxis_title": "age class",
    #         "yaxis_title": "",
    #         # "yaxis": {"range": [0, 1]},
    #     },
    # },
}
