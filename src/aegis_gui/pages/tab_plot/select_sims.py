import dash
from dash import callback, Output, Input, ALL, State, dcc

from aegis_gui.utilities import log
import dash_bootstrap_components as dbc


# def make_single_dropdown(dropdown_options):
#     # TODO plot initial plots
#     initial_value = dropdown_options[0] if dropdown_options else None
#     initial_value = None
#     return dcc.Dropdown(id="dropdown-single", options=dropdown_options, value=initial_value)


def make_multi_dropdown(dropdown_options, dropdown_value=[]):
    if dropdown_value == []:
        dropdown_value = [dropdown_options[0]] if dropdown_options else []
    assert isinstance(dropdown_value, list)
    return dbc.InputGroup(
        [
            # dbc.InputGroupText("Plotted simulation"),
            dcc.Dropdown(
                id="dropdown-multi",
                options=dropdown_options,
                value=dropdown_value,
                multi=True,
                placeholder="Select simulations to plot...",
                className="plot-dropdown",
                style={"width": "100%"},
                persistence=True,
                clearable=False,
            ),
        ]
    )


@callback(
    Output({"type": "loading-graph-div", "index": ALL}, "parent_style"),
    Input("figure-select", "value"),
    State({"type": "loading-graph-div", "index": ALL}, "id"),
    State({"type": "loading-graph-div", "index": ALL}, "parent_style"),
)
def change_visibility(figure_selected, graph_ids, parent_styles):

    new_parent_styles = []

    for graph_id, parent_style in zip(graph_ids, parent_styles):
        new_parent_style = parent_style.copy() if parent_style else {}
        should_be_invisible = graph_id["index"] != figure_selected

        if should_be_invisible:
            new_parent_style["display"] = "none"
        elif "display" in new_parent_style:
            del new_parent_style["display"]

        new_parent_styles.append(new_parent_style)

    return new_parent_styles
