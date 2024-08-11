from dash import callback, Output, Input, State, ALL, MATCH, ctx, html
from aegis_gui.utilities import log_funcs, utilities
import dash_bootstrap_components as dbc
import dash
import os
import shutil


def make(filename):
    return html.Div(
        [
            dbc.Button(
                [html.I(className="bi bi-trash3-fill"), "Delete"],
                id="delete-simulation-button",
                value=filename,
                color="danger",
                className="me-2",
            ),
            dash.dcc.Store("delete-simulation-filename", data=filename),
        ]
    )


# @callback(
#     Output({"type": "delete-simulation-button", "index": MATCH}, "disabled"),
#     Input({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
# )
# @log_funcs.log_debug
# def disable_delete_simulation_button(n_clicks):
#     if n_clicks is None:
#         return False
#     return True


@callback(
    Output("simlog-section-table", "children"),
    Input({"type": "delete-simulation-button", "index": ALL}, "n_clicks"),
    State("simlog-section-table", "children"),
    prevent_initial_call=True,
)
@log_funcs.log_info
def change_simlog(n_clicks, current):

    # If delete button triggered the action, delete the simulation
    if isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") == "delete-simulation-button":
        filename = ctx.triggered_id["index"]
        config_path = utilities.get_config_path(filename)
        sim_path = config_path.parent / filename
        remove_simulation_data(sim_path=sim_path, config_path=config_path)
        return current
    else:
        raise Exception


def delete_simulation(filename):
    config_path = utilities.get_config_path(filename)
    sim_path = config_path.parent / filename
    remove_simulation_data(sim_path=sim_path, config_path=config_path)


def remove_simulation_data(sim_path, config_path):
    """Removes the simulation directory and configuration file."""
    # Remove the directory and all its contents
    if os.path.isdir(sim_path):
        shutil.rmtree(sim_path)  # This removes the directory and all its contents

    # Remove the configuration file
    if os.path.isfile(config_path):
        os.remove(config_path)  # This removes the file


@callback(
    Output("modal-centered", "is_open"),
    dash.Output("sim-select", "options"),
    dash.Output("sim-select", "value"),
    Input("delete-simulation-button", "n_clicks"),
    Input("close-centered", "n_clicks"),
    Input("permanently-delete", "n_clicks"),
    State("delete-simulation-filename", "data"),
    dash.State("sim-select", "options"),
    prevent_initial_call=True,
)
def toggle_modal(n1, n2, n3, filename, current_options):
    if ctx.triggered_id == "delete-simulation-button" and n1 is not None:
        return True, dash.no_update, dash.no_update

    if ctx.triggered_id == "close-centered" and n2 is not None:
        return False, dash.no_update, dash.no_update

    if ctx.triggered_id == "permanently-delete" and n3 is not None:
        delete_simulation(filename=filename)
        new_options = [d for d in current_options if d["label"] != filename]
        return False, new_options, new_options[0]["value"] if new_options else ""

    # TODO issue when no new options

    return dash.no_update, dash.no_update, dash.no_update


def make_modal():
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Delete simulation"), close_button=True),
            dbc.ModalBody("Are you sure you want to permanently delete the simulation data?"),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Yes, delete",
                        id="permanently-delete",
                        n_clicks=0,
                        color="danger",
                        outline=True,
                    ),
                    dbc.Button(
                        "Close",
                        id="close-centered",
                        n_clicks=0,
                    ),
                ]
            ),
        ],
        id="modal-centered",
        centered=True,
        is_open=False,
    )
