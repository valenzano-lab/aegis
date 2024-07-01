from dash import callback, Output, Input, State, ALL, MATCH, ctx
import subprocess
import logging
from aegis.visor import utilities
from aegis.visor.pages.tab_simlog.utilities import make_table
import yaml

from aegis.utilities.container import Container


# SHOW SIMS
@callback(
    Output("simlog-section-table", "children"),
    Input({"type": "delete-simulation-button", "index": ALL}, "n_clicks"),
    # Input("simlog-view-button", "n_clicks"),
    Input("main-url", "pathname"),
    State({"type": "selection-state", "index": ALL}, "data"),
)
@utilities.log_debug
def show_sims(n_clicks1, pathname, data):
    print("yooooo", pathname)
    if ctx.triggered_id is None:  # If initial call
        return [make_table(selection_states={"default": True}, sim_data=None)]

    if ctx.triggered_id == "simlog-view-button":
        pass

    # If delete button triggered the action, delete the simulation
    if isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") == "delete-simulation-button":
        filename = ctx.triggered_id["index"]
        config_path = utilities.get_config_path(filename)
        sim_path = config_path.parent / filename
        subprocess.run(["rm", "-r", sim_path], check=True)
        subprocess.run(["rm", config_path], check=True)

    selection_states = {l[0]: l[1] for l in data}

    return [make_table(selection_states=selection_states, sim_data=None)]


@callback(
    Output({"type": "config-dcc-download", "index": MATCH}, "data"),
    Input({"type": "config-download-button", "index": MATCH}, "n_clicks"),
    State({"type": "config-download-basepath", "index": MATCH}, "children"),
)
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
def disable_delete_simulation_button(n_clicks):
    if n_clicks is None:
        return False
    return True


# CHOOSE
@callback(
    Output({"type": "selection-state", "index": MATCH}, "data"),
    Input({"type": "display-button", "index": MATCH}, "n_clicks"),
    State({"type": "selection-state", "index": MATCH}, "data"),
)
@utilities.log_debug
def click_display_button(n_clicks, data):
    if n_clicks is None:
        return data
    filename, selected = data
    selected = not selected
    return [filename, selected]


@callback(
    Output({"type": "display-button", "index": MATCH}, "className"),
    Input({"type": "selection-state", "index": MATCH}, "data"),
)
@utilities.log_debug
def stylize_display_buttons(data):
    selected = data[1]
    if selected:
        return "checklist checked"
    else:
        return "checklist"


# LOAD DATA WHEN SELECTED
# TODO not used but can be used to improve performance
@callback(
    Output({"type": "sim-data", "index": MATCH}, "data"),
    Input({"type": "selection-state", "index": MATCH}, "data"),
    State({"type": "sim-data", "index": MATCH}, "data"),
)
@utilities.log_debug
def load_data(selection_state, data):
    filename, selected = selection_state
    if selected and data is None:
        logging.info(f"Loading {filename} successful.")
    return data
