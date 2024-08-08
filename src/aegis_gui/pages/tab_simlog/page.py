import dash
from dash import html
from aegis_gui.utilities import log_funcs

from aegis_gui.pages.tab_simlog import dropdown

from aegis_gui.pages.tab_simlog.table import generate_initial_simlog


dash.register_page(__name__, path="/simlog", name="simlog")


PREFACE = [
    html.Div(
        children=[
            # TODO change text
            html.H1("Simulation log tab"),
            html.P(
                [
                    """Access simulation metadata, download configuration files and simulation data, and terminate or delete simulations."""
                    # """
                    # This is the list tab. Here you can see which simulations you have started and what their status is, including their metadata.
                    # You can also delete simulations and mark which simulations you would like to display in the plot tab.
                    # Note that the 'default' simulation cannot be deleted – it comes with the installation and can be used
                    # as an exploratory example. All parameters for the default simulation have default values.
                    # """,
                ],
                style={"margin-bottom": "2rem"},
            ),
        ],
    )
]


@log_funcs.log_debug
def layout(sim=None):
    return html.Div(
        id="simlog-section",
        children=PREFACE + [dropdown.make_select(selected=sim), dropdown.get_info_div()],
        # + [
        #     html.Div(id="simlog-section-table", children=generate_initial_simlog()),
        # table.make_table(),
        # ],
    )
