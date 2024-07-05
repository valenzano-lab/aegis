from dash import callback, Output, Input, State, ALL, MATCH, ctx
import subprocess
import logging
from aegis.visor.utilities import log_funcs, utilities
from aegis.visor.pages.tab_simlog.utilities import make_simlog_table
import yaml

from aegis.utilities.container import Container


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
        return [make_simlog_table()]
    else:
        raise Exception


@callback(
    Output({"type": "config-dcc-download", "index": MATCH}, "data"),
    Input({"type": "config-download-button", "index": MATCH}, "n_clicks"),
    State({"type": "config-download-basepath", "index": MATCH}, "children"),
)
@log_funcs.log_debug
def config_file_download_button(n_clicks, basepath):
    if n_clicks is None:
        return n_clicks

    container = Container(basepath)
    config = container.get_config()
    return {
        "content": yaml.dump(config),
        "filename": container.name + ".yml",
    }


@callback(
    Output({"type": "delete-simulation-button", "index": MATCH}, "disabled"),
    Input({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
)
@log_funcs.log_debug
def disable_delete_simulation_button(n_clicks):
    if n_clicks is None:
        return False
    return True
