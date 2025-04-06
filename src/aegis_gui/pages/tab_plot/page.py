import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from aegis_gui.utilities import log, utilities
from aegis_sim.utilities.container import Container
from aegis_gui.pages.tab_plot import select_graph, select_sims, graph, refresh
import aegis_gui.pages.tab_plot.draw  # important callback


dash.register_page(__name__, path="/plot", name="AEGIS | Plot")


def layout(sim=None):  # use function to ensure statelessness

    simnames = [Container(path).basepath.stem for path in utilities.get_sim_paths()]
    dropdown_options = sorted(simnames)

    if not dropdown_options:
        body = [dash.html.P("No simulations to display.")]
    else:
        body = [
            # TODO change text
            html.Div(
                id="plot-top-panel",
                style={"width": "100%"},
                children=[
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(select_graph.select, width=8),
                                    # dbc.Col(html.Label("(plotting function)"), width=3),
                                ],
                                className="g-0",
                                style={"marginBottom": "0.3rem"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        select_sims.make_multi_dropdown(
                                            dropdown_options=dropdown_options,
                                            dropdown_value=[sim] if sim is not None else [],
                                        ),
                                        width=8,
                                    ),
                                    # dbc.Col(html.Label("(plotted simulations)"), width=3),
                                ],
                                className="g-0",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(refresh.inputgroup),
                                ]
                            ),
                        ],
                        style={"margin": "2rem 0"},
                    ),
                ],
            ),
            html.Div(
                [
                    html.Div(children=graph.get_graphs(), id="plot-bottom-left-panel", style={"width": "auto"}),
                    html.Div(children=[], id="plot-bottom-right-panel", style={"marginLeft": "2rem"}),
                ],
                style={"width": "100%", "display": "flex"},
            ),
            # reload.make_reload_button(),
            # TODO the order of figures is set by FIG_SETUP; thats how the tabs are generated and how you should return them
            # dropdowns.make_single_dropdown(dropdown_options),
            # tabs.get_tabs_single_layout(),
            # dropdowns.make_multi_dropdown(dropdown_options),
            # tabs.get_tabs_multi_layout(),
        ]

    faq = [
        html.H3("Frequently Asked Questions", style={"marginTop": "2rem"}),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    """
                    Intrinsic trait refers to a hypothetical trait that is limited only by genetics, and would be observed in an ideal environment.
                    For example, intrinsic mortality refers to mortality that occurs regardless of environmental conditions (amount of food, infection hazards, etc.).
                    \n
                    Life history traits are generally understood as observable phenotypes such as death and reproduction events. 
                    However, when interpreting observed phenotypes, it is useful to distinguish between the observed and intrinsic levels of a specific trait.
                    For example, individuals of most species are more likely to die as they get older, which reflects some intrinsic state that deteriorates with age.
                    This intrinsic state is not directly observable, and can only be estimated from the simple binary observed state 'dead' or 'alive'.
                    """,
                    title="What is meant by 'intrinsic'?",
                ),
                dbc.AccordionItem(
                    """Life history traits arise from the interplay between intrinsic (genetic) and extrinsic (environmental) factors.
                    In evolutionary analyses, distinguishing these sources of variation is crucial: genetic components are heritable and thus subject to selection,
                    whereas environmental influences are not. Nevertheless, selection acts on the realised phenotype-the integrated outcome of both genetic architecture and
                    environmental context. While disentangling these effects is often infeasible in empirical systems, simulations offer a tractable means of doing so.
                    This partitioning enables the investigation of how evolutionary processes act on intrinsic traits to shape phenotypic outcomes.
                    Furthermore, life history traits are mostly observed as pointwise events, such as a reproductive event or a death; thus mortality and fertility
                    can only be meaningfully extracted from population-level data. In AEGIS, intrinsic traits are transparent to the user.
                    """,
                    title="Why distinguish 'intrinsic' and 'observed' traits?",
                ),
            ],
            start_collapsed=True,
        ),
    ]

    # TODO plot initially
    children = [html.Div(children=PREFACE + body + faq)]

    return html.Div(id="plot-section", children=children)


PREFACE = [
    html.H1("Plot tab"),
    html.P(
        """
                            Explore simulations and export figures.
                            """
        # This is the plot tab. Here you can explore the simulation visually.
        # You can also download the figures and the data used for plotting.
        # The figures are interactive – if multiple simulations are displayed, you can click on
        # simulation IDs to toggle their visibility; you can also zoom in and out.
        # For figures that show time-specific data, you can use sliders to change the time point plotted.
        # """
    ),
    # dropdowns.make_single_dropdown(dropdown_options),
]
