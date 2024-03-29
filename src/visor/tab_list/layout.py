from dash import html, dcc
from visor import funcs
import datetime
from aegis.help.container import Container


@funcs.log_debug
def make_table_row(
    selection_state, sim_data, log, input_summary, output_summary, basepath, filename
):
    if len(log) > 0:
        logline = log.iloc[-1].to_dict()
    else:
        logline = {"ETA": None, "stage": None, "stg/min": None}

    if output_summary is None:
        status = ["not finished", "not extinct"]
    elif output_summary["extinct"]:
        status = ["finished", "extinct"]
    else:
        status = ["finished", "not extinct"]

    time_of_creation = (
        datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime(
            "%Y-%m-%d %H:%M"
        )
        if input_summary
        else None
    )

    time_of_finishing = (
        datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime(
            "%Y-%m-%d %H:%M"
        )
        if output_summary
        else None
    )

    extinct = status[1]
    eta = logline["ETA"]

    row = html.Tr(
        id={"type": "list-table-row", "index": filename},
        children=[
            dcc.Store(
                {"type": "selection-state", "index": filename},
                data=(filename, selection_state),
            ),
            dcc.Store(
                {"type": "sim-data", "index": filename},
                data=sim_data,
            ),
            html.Td(
                filename,
                style={"padding-left": "1.3rem", "padding-right": "0.8rem"},
            ),
            html.Td(
                children=[
                    html.Button(
                        children="display",
                        id={
                            "type": "display-button",
                            "index": filename,
                        },
                        className="checklist checked"
                        if selection_state
                        else "checklist",
                    ),
                ]
            ),
            html.Td(html.P(time_of_creation)),
            html.Td(html.P(time_of_finishing)),
            html.Td(html.P(extinct)),
            html.Td(html.P(eta if time_of_finishing is None else "       ")),
            html.Td(html.P(str(basepath))),
            html.Td(
                html.Button(
                    "delete",
                    id={
                        "type": "delete-simulation-button",
                        "index": filename,
                    },
                    value=filename,
                )
                if filename != "default"
                else None,
                style={"padding-right": "1rem"},
            ),
        ],
    )
    return row


@funcs.log_debug
def make_table(selection_states={}, sim_data=None):
    # Get data from folders
    paths = funcs.get_sim_paths()

    # Make table
    # - specify table header
    table_header = html.Tr(
        [
            html.Th("ID", style={"padding-left": "1.3rem", "padding-right": "0.8rem"}),
            html.Th("DISPLAY"),
            html.Th("CREATED"),
            html.Th("FINISHED"),
            html.Th("EXTINCT"),
            html.Th("ETA"),
            html.Th("FILEPATH"),
            html.Th("DELETE", style={"padding-right": "1rem"}),
        ],
    )
    table = [table_header]

    # - add table rows
    for path in paths:
        container = Container(path)
        filename = container.basepath.stem
        selection_state = selection_states.get(filename, False)
        element = make_table_row(
            selection_state=selection_state,
            sim_data=sim_data,
            log=container.get_log(),
            input_summary=container.get_input_summary(),
            output_summary=container.get_output_summary(),
            basepath=container.basepath,
            filename=filename,
        )
        table.append(element)

    return html.Table(children=table, id="list-table")


@funcs.log_debug
def get_list_layout():
    # selection_states = [
    #     dcc.Store({"type": "selection-state", "index": sim}, data=False)
    #     for sim in funcs.get_sims()
    # ]
    return html.Div(
        id="list-section",
        style={"display": "none"},
        children=[
            # html.Div(id="selection-states", children=selection_states),
            html.Div(id="list-section-table", children=[]),
        ],
    )
