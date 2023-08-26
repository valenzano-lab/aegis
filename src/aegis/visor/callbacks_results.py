from dash import html, dcc, callback, Output, Input, State, ALL, MATCH, ctx
import datetime

from aegis.visor import funcs
from aegis.help.container import Container
import subprocess

SELECTION = set()


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

    SELECTION.remove(filename)

    subprocess.run(["rm", "-r", sim_path], check=True)
    subprocess.run(["rm", config_path], check=True)
    return "deleted"


# @callback(
#     Output("result-section", "children", allow_duplicate=True),
#     [Input("testo", "n_clicks")],
#     [State("result-section", "children")],
#     prevent_initial_call=True,
# )
# def test(_, children):
#     # children += html.P("asdf")
#     children.append(html.P("asdfjkwerj"))
#     return children


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
        # html.Button(id="testo", children="testo"),
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
        ),
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
            children=[
                html.Td(container.basepath.stem),
                html.Td(
                    dcc.Checklist(
                        id={
                            "type": "result-checklist",
                            "index": container.basepath.stem,
                        },
                        options=[{"label": "", "value": "y"}],
                        value=["y"] if container.basepath.stem in SELECTION else [],
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
                        "delete",
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
    Output({"type": "result-checklist", "index": MATCH}, "style"),
    Input({"type": "result-checklist", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def update_selection(value):

    sim = ctx.triggered_id["index"]
    if value == ["y"]:
        SELECTION.add(sim)
    else:
        SELECTION.remove(sim)
    print(SELECTION)
    return {}
