from dash import callback, Output, Input, State, ALL, MATCH, ctx
import subprocess
import logging
from visor import funcs
from visor.tab_list.layout import make_table


# SHOW SIMS
@callback(
    Output("list-section-table", "children"),
    Input({"type": "delete-simulation-button", "index": ALL}, "n_clicks"),
    Input("list-view-button", "n_clicks"),
    State({"type": "selection-state", "index": ALL}, "data"),
    prevent_initial_call=True,
)
@funcs.log_info
def show_sims(n_clicks1, n_clicks2, data):
    if ctx.triggered_id is None:
        return []

    if ctx.triggered_id == "list-view-button":
        pass

    # If delete button triggered the action, delete the simulation
    if (
        isinstance(ctx.triggered_id, dict)
        and ctx.triggered_id.get("type") == "delete-simulation-button"
    ):
        filename = ctx.triggered_id["index"]
        config_path = funcs.get_config_path(filename)
        sim_path = config_path.parent / filename
        subprocess.run(["rm", "-r", sim_path], check=True)
        subprocess.run(["rm", config_path], check=True)

    selection_states = {l[0]: l[1] for l in data}

    return [make_table(selection_states=selection_states, sim_data=None)]


# CHOOSE
@callback(
    Output({"type": "selection-state", "index": MATCH}, "data"),
    Input({"type": "display-button", "index": MATCH}, "n_clicks"),
    State({"type": "selection-state", "index": MATCH}, "data"),
)
@funcs.log_info
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
@funcs.log_info
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
@funcs.log_info
def load_data(selection_state, data):
    filename, selected = selection_state
    if selected and data is None:
        logging.info(f"Loading {filename} successful!")
    return data
