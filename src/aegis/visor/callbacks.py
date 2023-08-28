from dash import Dash, html, dcc, callback, Output, Input, State, ALL, MATCH
import dash

import logging
import datetime

from aegis.visor import funcs
from aegis.help.container import Container
import subprocess


@callback(
    Output("figure-section", "style"),
    Output("sim-section", "style"),
    Output("result-section", "style"),
    Input("plot-view-button", "n_clicks"),
    Input("config-view-button", "n_clicks"),
    Input("result-view-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def toggle_display(*_):
    triggered = dash.callback_context.triggered[0]["prop_id"].split("-")[0]
    styles = {
        "plot": [{"display": "block"}, {"display": "none"}, {"display": "none"}],
        "config": [{"display": "none"}, {"display": "block"}, {"display": "none"}],
        "result": [{"display": "none"}, {"display": "none"}, {"display": "block"}],
    }
    return styles[triggered]


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

    subprocess.run(["rm", "-r", sim_path], check=True)
    subprocess.run(["rm", config_path], check=True)
    return "deleted"


@callback(
    Output("result-section", "children"),
    Input("result-view-button", "n_clicks"),
    Input({"type": "delete-simulation-button", "index": ALL}, "children"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def refresh_result_section(*_):

    paths = funcs.get_sim_paths()
    containers = [Container(path) for path in paths]
    table_elements = [
        html.Tr(
            [
                html.Th("ID"),
                html.Th("DISPLAY"),
                html.Th("CREATED"),
                # html.Th("edited"),
                html.Th("RUNNING STATUS"),
                html.Th("EXTINCT STATUS"),
                html.Th("TIME REMAINING"),
                # html.Th("stage"),
                html.Th("STAGE PER MINUTE"),
                html.Th("FILEPATH"),
                html.Th("DELETE"),
            ],
        )
    ]

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
        # time_of_edit = datetime.datetime.fromtimestamp(
        #     container.paths["log"].stat().st_mtime
        # )

        element = html.Tr(
            [
                html.Td(container.basepath.stem),
                html.Td(
                    dcc.Checklist(
                        id=str(container.basepath),
                        options=[{"label": "", "value": "yes"}],
                        value=[],
                    ),
                ),
                html.Td(html.P(time_of_creation)),
                # date created
                # html.Td(html.P(time_of_edit)),
                html.Td(html.P(status[0])),
                html.Td(html.P(status[1])),
                html.Td(html.P(logline["ETA"])),
                # html.Td(html.P(logline["stage"])),
                html.Td(html.P(logline["stg/min"])),
                html.Td(html.P(str(container.basepath))),
                html.Td(
                    html.Button(
                        "delete simulation",
                        id={
                            "type": "delete-simulation-button",
                            "index": container.basepath.stem,
                        },
                        value=container.basepath.stem,
                    )
                ),
            ],
        )
        table_elements.append(element)

    return html.Table(children=table_elements, className="result-table")


@callback(
    [
        Output("dynamic-dropdown", "options"),
        Output("dynamic-dropdown", "value"),
    ],
    [
        Input("plot-view-button", "n_clicks"),
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


@callback(
    Output("simulation-run-button", "disabled"),
    Input("config-make-text", "value"),
)
def block_sim_button(filename):

    if filename is None or filename == "":
        return True

    sim_exists = funcs.sim_exists(filename)
    if sim_exists:
        return True
    return False
