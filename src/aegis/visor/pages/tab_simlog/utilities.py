from dash import html, dcc, dash_table
from aegis.visor.utilities import log_funcs, utilities
import datetime
from aegis.utilities.container import Container

from aegis.visor.config import config

from . import zip_button, folder_size


@log_funcs.log_debug
def make_simlog_table():
    paths = utilities.get_sim_paths()

    # Make table
    # - specify table header
    table_header = html.Tr(
        [
            html.Th("ID", style={"padding-left": "1.3rem", "padding-right": "0.8rem"}),
            html.Th("CREATED"),
            html.Th("FINISHED"),
            html.Th("EXTINCT"),
            html.Th("RUNNING"),
            html.Th("SIZE"),
            html.Th("ETA"),
            html.Th("FILEPATH"),
            html.Th("CONFIG FILE"),
            html.Th("DELETE", style={"padding-right": "1rem"}),
            html.Th("ZIP DATA"),
        ],
    )
    table = [table_header]

    # - add table rows
    for path in paths:
        container = Container(path)
        filename = container.basepath.stem
        element = make_table_row(
            log=container.get_log(),
            input_summary=container.get_input_summary(),
            output_summary=container.get_output_summary(),
            basepath=container.basepath,
            filename=filename,
            ticker_stopped=container.has_ticker_stopped(),
        )
        table.append(element)

    return html.Table(children=table, id="simlog-table")


# @log_funcs.log_debug
def make_table_row(log, input_summary, output_summary, basepath, filename, ticker_stopped):
    if len(log) > 0:
        logline = log.iloc[-1].to_dict()
    else:
        logline = {"ETA": None, "step": None, "stg/min": None}

    if not output_summary:
        status = ["not finished", "not extinct"]
    elif output_summary["extinct"]:
        status = ["finished", "extinct"]
    else:
        status = ["finished", "not extinct"]

    time_of_creation = (
        datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime("%Y-%m-%d %H:%M")
        if input_summary
        else None
    )

    time_of_finishing = (
        datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime("%Y-%m-%d %H:%M")
        if output_summary
        else None
    )

    extinct = status[1]
    running = "no" if ticker_stopped else "yes"
    eta = logline["ETA"]

    row = html.Tr(
        id={"type": "simlog-table-row", "index": filename},
        children=[
            html.Td(
                filename,
                style={"padding-left": "1.3rem", "padding-right": "0.8rem"},
            ),
            html.Td(html.P(time_of_creation)),
            html.Td(html.P(time_of_finishing)),
            html.Td(html.P(extinct)),
            html.Td(html.P(running)),
            folder_size.get_simlog_layout(basepath),
            html.Td(html.P(eta if time_of_finishing is None else "       ")),
            html.Td(html.P(str(basepath), id={"type": "config-download-basepath", "index": filename})),
            html.Td(
                children=[
                    html.Button(
                        "download",
                        id={"type": "config-download-button", "index": filename},
                        value=filename,
                        # style={"padding-right": "1rem"},
                    ),
                    dcc.Download(id={"type": "config-dcc-download", "index": filename}),
                ],
            ),
            html.Td(
                (
                    html.Button(
                        "delete",
                        id={"type": "delete-simulation-button", "index": filename},
                        value=filename,
                    )
                    if ~config.can_delete_default_sim or filename != "default"
                    else None
                ),
                style={"padding-right": "1rem"},
            ),
            zip_button.get_zip_button_layout(filename=filename),
        ],
    )
    return row


def generate_initial_simlog():
    return [make_simlog_table()]
