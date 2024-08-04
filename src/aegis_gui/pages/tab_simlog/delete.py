from dash import callback, Output, Input, State, ALL, MATCH, ctx, html
import subprocess
from aegis_gui.utilities import log_funcs, utilities
import dash_bootstrap_components as dbc
import dash


def make(filename):
    return html.Div(
        [
            dbc.Button(
                [html.I(className="bi bi-trash3-fill"), "delete"],
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
        subprocess.run(["rm", "-r", sim_path], check=True)
        subprocess.run(["rm", config_path], check=True)
        return current
    else:
        raise Exception


def delete_simulation(filename):
    config_path = utilities.get_config_path(filename)
    sim_path = config_path.parent / filename
    subprocess.run(["rm", "-r", sim_path], check=True)
    subprocess.run(["rm", config_path], check=True)


@callback(
    Output("modal-centered", "is_open"),
    Input("delete-simulation-button", "n_clicks"),
    Input("close-centered", "n_clicks"),
    Input("permanently-delete", "n_clicks"),
    State("delete-simulation-filename", "data"),
    prevent_initial_call=True,
)
def toggle_modal(n1, n2, n3, filename):

    if ctx.triggered_id == "close-centered":
        return False

    if ctx.triggered_id == "delete-simulation-button":
        return True

    if ctx.triggered_id == "permanently-delete":
        delete_simulation(filename=filename)
        return False


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
