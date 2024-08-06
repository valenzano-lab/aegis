import dash
from dash import html
import dash_bootstrap_components as dbc
from aegis.utilities.container import Container
from aegis_gui.utilities import utilities
import pathlib
from aegis_gui.pages.tab_simlog import zip, delete, terminate, download
from aegis.utilities.get_folder_size import get_folder_size_with_du


def make_multi_dropdown(selected=None):

    paths = utilities.get_sim_paths()

    if selected:
        for path in paths:
            if Container(path).name == selected:
                selected_path = path
                break
    else:
        selected_path = paths[0]

    return dbc.Select(
        id="sim_select",
        options=[{"label": path.stem, "value": str(path)} for path in paths],
        # value=["default"] if "default" in dropdown_options else [],
        value=str(selected_path),
        placeholder="Select simulations to plot...",
        # className="plot-dropdown",
        # multiple=True,
        style={"margin-bottom": "1rem"},
    )


# Div to display simulation information
def get_info_div():
    return dash.html.Div(children=[], id="sim_info_div")


# Function to get simulation information
def get_sim_info(path):
    container = Container(path)
    return {
        "log": container.get_log(),
        "is": container.get_input_summary(),
        "output_summary": container.get_output_summary(),
        "basepath": str(container.basepath),
        "filename": container.basepath.stem,
        "ticker_stopped": container.has_ticker_stopped(),
    }


# Callback to update sim_info_div
@dash.callback(
    dash.Output("sim_info_div", "children"),
    dash.Input("sim_select", "value"),
)
def update_info_div(selected_path):
    if not selected_path:
        return html.Div("No simulation selected.")

    path = pathlib.Path(selected_path)
    sim_info = get_sim_info(path)

    # Create list items for each item in sim_info
    list_items = []
    for key, value in sim_info.items():
        list_items.append(dbc.ListGroupItem([html.Strong(f"{key}: "), html.Span(str(value))]))

    list_items.append(dbc.ListGroupItem([get_folder_size_with_du(path)]))
    list_items.append(
        dbc.ListGroupItem([html.P(str(path), id={"type": "config-download-basepath", "index": path.stem})])
    )
    list_items.append(
        dbc.ListGroupItem(
            [
                delete.make(path.stem),
                delete.make_modal(),
                terminate.make_button(filename=path.stem),
                zip.get_zip_button_layout(filename=path.stem),
                download.make_button(path.stem),
                download.make_dcc(path.stem),
            ],
            style={"display": "flex"},
        )
    )

    # Return a dbc.ListGroup with the simulation information
    return dbc.ListGroup(list_items, flush=True)