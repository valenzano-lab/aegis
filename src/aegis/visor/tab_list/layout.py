from dash import html, dcc
from aegis.visor import funcs
import datetime
from aegis.help.container import Container


@funcs.print_function_name
def make_table_row(container, selection_state):
    if len(container.get_log()) > 0:
        logline = container.get_log().iloc[-1].to_dict()
    else:
        logline = {"ETA": None, "stage": None, "stg/min": None}
    input_summary = container.get_input_summary()
    output_summary = container.get_output_summary()

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

    filename = container.basepath.stem
    basepath = container.basepath
    extinct = status[1]
    eta = logline["ETA"]

    row = html.Tr(
        id={"type": "list-table-row", "index": filename},
        children=[
            dcc.Store(
                {"type": "selection-state", "index": filename},
                data=(filename, selection_state),
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
                ),
                style={"padding-right": "1rem"},
            ),
        ],
    )
    return row


@funcs.print_function_name
def make_table(selection_states={}):
    # Get data from folders
    paths = funcs.get_sim_paths()
    containers = [Container(path) for path in paths]

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
    for container in containers:
        filename = container.basepath.stem
        selection_state = selection_states.get(filename, False)
        print(selection_state)
        element = make_table_row(container, selection_state=selection_state)
        table.append(element)

    return html.Table(children=table, id="list-table")


@funcs.print_function_name
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
