import dash
from dash import html, dcc
from aegis.visor.utilities import log_funcs, utilities
from aegis.utilities.container import Container
from aegis.visor.pages.tab_plot import dropdowns, tabs, replot_button


dash.register_page(__name__, path="/plot", name="plot")


@log_funcs.log_debug
def layout():  # use function to ensure statelessness

    simnames = [Container(path).basepath.stem for path in utilities.get_sim_paths()]
    dropdown_options = [simname for simname in simnames]
    # TODO plot initially
    children = [
        html.Div(
            children=[
                # TODO change text
                html.P(
                    [
                        """
                        This is the plot tab. Here you can explore the simulation visually.
                        You can also download the figures and the data used for plotting.
                        The figures are interactive – if multiple simulations are displayed, you can click on
                        simulation IDs to toggle their visibility; you can also zoom in and out.
                        For figures that show time-specific data, you can use sliders to change the time point plotted.
                        """,
                    ],
                    style={"margin-bottom": "2rem"},
                ),
                replot_button.make_reload_button(),
                # TODO the order of figures is set by FIG_SETUP; thats how the tabs are generated and how you should return them
                dropdowns.make_single_dropdown(dropdown_options),
                tabs.get_tabs_single_layout(),
                dropdowns.make_multi_dropdown(dropdown_options),
                tabs.get_tabs_multi_layout(),
            ],
        )
    ]

    return html.Div(id="plot-section", children=children)
