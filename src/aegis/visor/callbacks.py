from dash import Dash, html, dcc, callback, Output, Input, State, ALL, MATCH

import logging
import datetime

from aegis.visor import funcs
from aegis.help.container import Container
import subprocess


@callback(
    Output("sim-section", "style"),
    Output("result-section", "style"),
    Output("figure-section", "style"),
    Input("toggle-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def toggle_display(n_clicks):
    # three possible displays
    styles = [{"display": "none"}] * 3
    styles[n_clicks % 3] = {"display": "block"}
    return styles[0], styles[1], styles[2]


@callback(
    Output({"type": "delete-simulation-button", "index": MATCH}, "children"),
    Input({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
    State({"type": "delete-simulation-button", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def delete_simulation(_, filename):
    config_path = funcs.get_config_path(filename)
    sim_path = config_path.parent / filename
    subprocess.run(f"rm -r {sim_path}", shell=True, check=True)
    subprocess.run(f"rm {config_path}", shell=True, check=True)
    return "deleted"


@callback(
    Output("result-section", "children"),
    Input("toggle-button", "n_clicks"),
    Input({"type": "delete-simulation-button", "index": ALL}, "children"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def refresh_result_section(*_):

    paths = funcs.get_sim_paths()
    print(paths)
    containers = [Container(path) for path in paths]
    elements = []

    for container in containers:
        if len(container.get_log()) > 0:
            logline = container.get_log().iloc[-1].to_dict()
        else:
            logline = {"ETA": None, "stage": None, "stg/min": None}
        input_summary = container.get_input_summary()
        output_summary = container.get_output_summary()

        if output_summary is None:
            status = ["running", "not extinct"]
        elif output_summary["extinct"]:
            status = ["finished", "extinct"]
        else:
            status = ["finished", "not extinct"]

        time_of_creation = (
            datetime.datetime.fromtimestamp(input_summary["time_start"])
            if input_summary
            else None
        )
        time_of_edit = datetime.datetime.fromtimestamp(
            container.paths["log"].stat().st_mtime
        )

        element = html.Div(
            children=[
                dcc.Checklist(
                    id=str(container.basepath),
                    options=[{"label": str(container.basepath.stem), "value": "yes"}],
                    value=[],
                ),
                # date created
                html.P(time_of_creation),
                html.P(time_of_edit),
                html.P(status[0]),
                html.P(status[1]),
                html.P(logline["ETA"]),
                html.P(logline["stage"]),
                html.P(logline["stg/min"]),
                html.P(str(container.basepath)),
                html.Button(
                    "delete simulation",
                    id={
                        "type": "delete-simulation-button",
                        "index": container.basepath.stem,
                    },
                    value=container.basepath.stem,
                ),
            ],
            style={"display": "block"},
        )
        elements.append(element)

    return elements


@callback(
    [
        Output("dynamic-dropdown", "options"),
        Output("dynamic-dropdown", "value"),
    ],
    [
        Input("toggle-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
@funcs.print_function_name
def refresh_dropdown_options(*_):
    paths = funcs.get_sim_paths()
    dropdown_options = [
        {"label": f"{i}. {str(path.stem)} ({str(path)})", "value": str(path)}
        for i, path in enumerate(paths)
    ]
    # BUG fix if no dropdown_options available
    return dropdown_options, dropdown_options[0]["value"]


@callback(
    Output("simulation-run-text", "children"),
    Input("simulation-run-button", "n_clicks"),
    State("config-make-text", "value"),
    [
        State(f"config-{k}", "value")
        for k, v in funcs.DEFAULT_CONFIG_DICT.items()
        if not isinstance(v, list)
    ],
    prevent_initial_call=True,
)
@funcs.print_function_name
def run_simulation(n_clicks, filename, *values):
    if n_clicks is None:
        return

    if filename is None or filename == "":
        logging.error("no filename given")
        return "no filename given. no simulation started."

    sim_exists = funcs.sim_exists(filename)
    if sim_exists:
        logging.error("sim with that name already exists")
        return "sim with that name already exists. no simulation started."

    # make config file
    custom_config = {
        k: val
        for (k, v), val in zip(funcs.DEFAULT_CONFIG_DICT.items(), values)
        if not isinstance(v, list)
    }
    funcs.make_config_file(filename, custom_config)

    # run simulation
    funcs.run(filename)


# @callback(
#     Output("config-make-button", "style"),
#     Input("config-make-button", "n_clicks"),
#     State("config-make-text", "value"),
#     [
#         State(f"config-{k}", "value")
#         for k, v in funcs.DEFAULT_CONFIG_DICT.items()
#         if not isinstance(v, list)
#     ],
#     prevent_initial_call=True,
# )
# @funcs.print_function_name
# def make_config_file(n_clicks, filename, *values):

#     custom_config = {
#         k: val
#         for (k, v), val in zip(funcs.DEFAULT_CONFIG_DICT.items(), values)
#         if not isinstance(v, list)
#     }

#     if n_clicks is not None:
#         funcs.make_config_file(filename, custom_config)
#     return {}
