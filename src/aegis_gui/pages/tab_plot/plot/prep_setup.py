import dash
import dash_bootstrap_components as dbc
from aegis_gui.pages.tab_plot.plot import prep_y
from aegis_gui.pages.tab_plot.plot import prep_x

AGGREGATION_BADGES = {
    "Population average": "Values shown in the plot are averaged over all individuals alive at the time of recording.",
    "Population median": "Values shown in the plot are medians among the individuals alive at the time of recording.",
    "Interval average": "Values shown in the plot are averages over a number of recordings made in a given interval.",
    "Interval median": "Values shown in the plot are medians among a number of recordings made in a given interval.",
}

FIG_SETUP = {
    "life table": {
        "title": "life table",
        "supports_multi": True,
        "prep_y": prep_y.get_life_table,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_bar_figure_not_stacked",
        "description": dash.dcc.Markdown(
            """
            Age structure of the population.
            \n
            """,
            mathjax=True,
        ),
        "aggregation": ["Interval average"],
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "proportion of living individuals",
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
            """,
            mathjax=True,
        ),
        "aggregation": ["Population average", "Interval average"],
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
            """,
            mathjax=True,
        ),
        "aggregation": ["Population median"],
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "intrinsic survivorship",
        },
    },
    "death table": {
        "title": "death table",
        "supports_multi": False,
        "prep_y": prep_y.get_death_table,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_bar_figure_stacked",
        "description": dash.dcc.Markdown(
            """
            Number of deaths per age class, stratified by cause of death.
            \n
            """,
            mathjax=True,
        ),
        "aggregation": ["Interval average"],
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "number of deaths",
        },
    },
    "death table normalized": {
        "title": "death table (normalized)",
        "supports_multi": False,
        "prep_y": prep_y.get_death_table_normalized,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_bar_figure_stacked",
        "description": dash.dcc.Markdown(
            """
            Number of deaths per age class, normalized and stratified by cause of death.
            """,
            mathjax=True,
        ),
        "aggregation": ["Interval average"],
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "number of deaths",
            "yaxis": {"range": [0, 1]},
        },
    },
    "intrinsic mortality": {
        "title": "intrinsic mortality",
        "supports_multi": True,
        "prep_y": prep_y.get_mortality_intrinsic,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            Genetic (individual-specific, heritable) mortality at a given age.
            This mortality is experienced by the individuals regardless of the environmental conditions.
            """,
            mathjax=True,
        ),
        "aggregation": ["Population median"],
        # dash.html.Span(
        #     "tooltips", id="tooltip-target-1", style={"textDecoration": "underline", "cursor": "pointer"}
        # ),
        # dbc.Tooltip("This is the first tooltip", target="tooltip-target-1"),
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
            Observed (individual-specific, heritable) mortality at a given age.
            In contrast to intrinsic mortality, in this measure all sources of mortality are considered.
            \n
            Missing data points are for age classes at which no living individuals have been observed, so no mortality
            can be computed.
            """,
            mathjax=True,
        ),
        "aggregation": ["Population average", "Interval average"],
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "total mortality",
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
            Expected lifespan at birth.
            \n
            Plotted over the course of the simulation.
            """,
            mathjax=True,
        ),
        "aggregation": ["Population average"],
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "life expectancy",
        },
    },
    "birth table": {
        "title": "birth table",
        "supports_multi": True,
        "prep_y": prep_y.get_birth_table,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_bar_figure_not_stacked",
        "description": dash.dcc.Markdown(
            """
            The number of newborns produced by parents of a given age class.
            \n
            """,
            mathjax=True,
        ),
        "aggregation": [],
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "number of newborns",
        },
    },
    "observed fertility": {
        "title": "observed fertility",
        "supports_multi": True,
        "prep_y": prep_y.get_fertility,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            The observed probability of producing offspring at each age class.
            """,
            mathjax=True,
        ),
        "aggregation": ["Population median"],
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "observed fertility",
        },
    },
    "intrinsic fertility": {
        "title": "intrinsic fertility",
        "supports_multi": True,
        "prep_y": prep_y.get_fertility_intrinsic,
        "prep_x": prep_x.get_ages,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            """
            Genetic (individual-specific, heritable) probability of producing offspring at a given age class.
            """,
            mathjax=True,
        ),
        "aggregation": ["Population median"],
        # dash.html.Span(
        #     "tooltips", id="tooltip-target-1", style={"textDecoration": "underline", "cursor": "pointer"}
        # ),
        # dbc.Tooltip("This is the first tooltip", target="tooltip-target-1"),
        # graph
        "figure_layout": {
            "xaxis_title": "age class",
            "yaxis_title": "intrinsic fertility",
        },
    },
    "population size": {
        "title": "population size",
        "supports_multi": True,
        "prep_y": prep_y.get_population_size_after_reproduction,
        "prep_x": prep_x.get_steps_non_multiplied,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            # TODO check this
            """
            Number of living individuals, counted after the reproduction phase.
            \n
            Plotted over the course of the simulation.
            """,
            mathjax=True,
        ),
        "aggregation": [],
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "population size",
        },
    },
    "egg number": {
        "title": "egg number",
        "supports_multi": True,
        "prep_y": prep_y.get_egg_number_after_reproduction,
        "prep_x": prep_x.get_steps_non_multiplied,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            # TODO check this
            """
            Number of eggs produced and not yet hatched and added to the population of living individuals.
            \n
            Plotted over the course of the simulation.
            """,
            mathjax=True,
        ),
        "aggregation": [],
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "egg number",
        },
    },
    "resource amount before scavenging": {
        "title": "resource amount before scavenging",
        "supports_multi": True,
        "prep_y": prep_y.get_resource_amount_before_scavenging,
        "prep_x": prep_x.get_steps_non_multiplied,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            # TODO check this
            """
            Amount of available resources before scavenging.
            \n
            Plotted over the course of the simulation.
            """,
            mathjax=True,
        ),
        "aggregation": [],
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "resource amount",
        },
    },
    "resource amount after scavenging": {
        "title": "resource amount after scavenging",
        "supports_multi": True,
        "prep_y": prep_y.get_resource_amount_after_scavenging,
        "prep_x": prep_x.get_steps_non_multiplied,
        "prep_figure": "make_line_figure",
        "description": dash.dcc.Markdown(
            # TODO check this
            """
            Amount of available resources after scavenging.
            \n
            Plotted over the course of the simulation.
            """,
            mathjax=True,
        ),
        "aggregation": [],
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "resource amount",
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
    "bit states": {
        "title": "bit states",
        "supports_multi": False,
        "prep_y": prep_y.get_bit_states,
        "prep_x": prep_x.get_steps_multiplied,
        "prep_figure": "make_heatmap_figure",
        "description": dash.dcc.Markdown(
            """
            Average bit states for each pseudogenomic site at a given simulation step.
            A pseudogenomic site can be either in a 0 or a 1 state.
            Each column represents an average pseudogenome.
            \n
            Note that results from only one simulation (the first from the selection) can be plotted at a single time.
            """,
            mathjax=True,
        ),
        "aggregation": ["Population average", "Interval average"],
        # graph
        "figure_layout": {
            "xaxis_title": "simulation step",
            "yaxis_title": "genome site",
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
            Frequency of alleles whose state is flipped in comparison to the ancestral state.
            Here, the ancestral state is the most frequent state in the previous genome record.
            Sites with derived allele frequency of 0 are ignored.
            """,
            mathjax=True,
        ),
        "aggregation": [],
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
    # v3: Muller plot (clonal expansion / contraction of founder lineages over time)
    # Active only when LINEAGE_TRACING=True and LINEAGE_RATE>0. Falls back to an
    # empty figure with an explanatory message if those weren't set.
    "muller (clonal dynamics)": {
        "title": "muller (clonal dynamics)",
        "supports_multi": False,
        "prep_y": prep_y.get_lineage_muller,
        "prep_x": prep_x.get_none,
        "prep_figure": "make_stacked_area_muller",
        "description": dash.dcc.Markdown(
            """
            Stacked-area frequency of founder lineages over time. Each band is one
            founder (an initial-population member); band thickness = number of
            living descendants at that step.
            \n\nRequires `LINEAGE_TRACING: true` and `LINEAGE_RATE > 0` in the
            simulation config. Asexual reproduction only.
            """,
            mathjax=True,
        ),
        "aggregation": [],
        "figure_layout": {
            "xaxis_title": "step",
            "yaxis_title": "relative frequency",
        },
    },
    # v3: Selection-coefficient experiment — allele frequency over time at the
    # ALLELE_INJECTION locus. Falls back to an empty figure when no log exists.
    "allele frequency trajectory": {
        "title": "allele frequency trajectory",
        "supports_multi": True,
        "prep_y": prep_y.get_allele_freq_trajectory,
        "prep_x": prep_x.get_steps_non_multiplied,
        "prep_figure": "make_allele_freq_trajectory",
        "description": dash.dcc.Markdown(
            """
            Frequency of the injected allele over time. Slope (after the injection
            step) under additive selection is the selection coefficient *s*; use
            `runs/fit_s.py` for the regression.
            \n\nRequires `ALLELE_INJECTION_STEP > 0` in the simulation config.
            """,
            mathjax=True,
        ),
        "aggregation": [],
        "figure_layout": {
            "xaxis_title": "step",
            "yaxis_title": "allele frequency",
        },
    },
}
