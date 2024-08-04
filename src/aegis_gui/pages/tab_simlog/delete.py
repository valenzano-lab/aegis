from dash import callback, Output, Input, State, ALL, MATCH, ctx, html
import subprocess
from aegis_gui.utilities import log_funcs, utilities
import dash_bootstrap_components as dbc


def make(filename):
    return dbc.Button(
        [html.I(className="bi bi-trash3-fill"), "delete"],
        id={"type": "delete-simulation-button", "index": filename},
        value=filename,
    )


@callback(
    Output({"type": "delete-simulation-button", "index": MATCH}, "disabled"),
    Input({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
)
@log_funcs.log_debug
def disable_delete_simulation_button(n_clicks):
    if n_clicks is None:
        return False
    return True


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
