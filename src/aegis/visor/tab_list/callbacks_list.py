from dash import html, dcc, callback, Output, Input, State, ALL, MATCH, ctx

from aegis.visor import funcs
import subprocess

from aegis.visor.tab_list.layout import make_table

SELECTION = set()


# # DELETE
# @callback(
#     Output({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
#     # Output("selection-states", "children"),
#     Input({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
#     State({"type": "delete-simulation-button", "index": MATCH}, "value"),
#     # State({"type": "selection-state", "index": ALL}, "data"),
# )
# @funcs.print_function_name
# def delete_simulation_data(n_clicks, filename):
#     if n_clicks is None:
#         return

#     # Delete simulation data
#     config_path = funcs.get_config_path(filename)
#     sim_path = config_path.parent / filename
#     subprocess.run(["rm", "-r", sim_path], check=True)
#     subprocess.run(["rm", config_path], check=True)

#     return n_clicks

#     # # Create new dcc.Stores
#     # dccs = [
#     #     dcc.Store({"type": "selection-state", "index": sim}, data=selection_state)
#     #     for sim, selection_state in zip(funcs.get_sims(), selection_states)
#     #     if sim != filename
#     # ]


# SHOW SIMS
@callback(
    Output("list-section-table", "children"),
    Input({"type": "delete-simulation-button", "index": ALL}, "n_clicks"),
    Input("list-view-button", "n_clicks"),
    State({"type": "selection-state", "index": ALL}, "data"),
    prevent_initial_call=True,
)
@funcs.print_function_name
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

    return [make_table(selection_states=selection_states)]

    # if children is None or children == []:
    #     children = [make_table()]
    # return children


# DELETE
# @callback(
#     # Output({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
#     Output("list-table", "children"),
#     Input({"type": "delete-simulation-button", "index": ALL}, "n_clicks"),
#     # Input({"type": "delete-simulation-button", "index": MATCH}, "n_clicks"),
#     State("list-table", "children"),
#     prevent_initial_call=True,
# )
# @funcs.print_function_name
# def click_delete_button(_, children):
#     if ctx.triggered_id is None:
#         return children
#     filename = ctx.triggered_id["index"]
#     children = [
#         child
#         for child in children
#         if "id" not in child["props"]  # header row
#         or child["props"]["id"]["index"] != filename  # not clicked row
#     ]

#     # Delete simulation data
#     try:
#         config_path = funcs.get_config_path(filename)
#         sim_path = config_path.parent / filename
#         subprocess.run(["rm", "-r", sim_path], check=True)
#         subprocess.run(["rm", config_path], check=True)
#     except:
#         print("Cannot delete")

#     return children


# CHOOSE
@callback(
    Output({"type": "selection-state", "index": MATCH}, "data"),
    Input({"type": "display-button", "index": MATCH}, "n_clicks"),
    State({"type": "selection-state", "index": MATCH}, "data"),
)
@funcs.print_function_name
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
@funcs.print_function_name
def stylize_display_buttons(data):
    selected = data[1]
    if selected:
        return "checklist checked"
    else:
        return "checklist"


# def make_table_row(container):
#     if len(container.get_log()) > 0:
#         logline = container.get_log().iloc[-1].to_dict()
#     else:
#         logline = {"ETA": None, "stage": None, "stg/min": None}
#     input_summary = container.get_input_summary()
#     output_summary = container.get_output_summary()

#     if output_summary is None:
#         status = ["not finished", "not extinct"]
#     elif output_summary["extinct"]:
#         status = ["finished", "extinct"]
#     else:
#         status = ["finished", "not extinct"]

#     time_of_creation = (
#         datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime(
#             "%Y-%m-%d %H:%M"
#         )
#         if input_summary
#         else None
#     )

#     time_of_finishing = (
#         datetime.datetime.fromtimestamp(input_summary["time_start"]).strftime(
#             "%Y-%m-%d %H:%M"
#         )
#         if output_summary
#         else None
#     )

#     filename = container.basepath.stem
#     basepath = container.basepath
#     extinct = status[1]
#     eta = logline["ETA"]
#     selected = False

#     row = html.Tr(
#         children=[
#             dcc.Store(
#                 {"type": "selection-state", "index": filename},
#                 data=(filename, selected),
#             ),
#             html.Td(
#                 filename,
#                 style={"padding-left": "1.3rem", "padding-right": "0.8rem"},
#             ),
#             html.Td(
#                 children=[
#                     html.Button(
#                         children="display",
#                         id={
#                             "type": "display-button",
#                             "index": filename,
#                         },
#                     ),
#                 ]
#             ),
#             html.Td(html.P(time_of_creation)),
#             html.Td(html.P(time_of_finishing)),
#             html.Td(html.P(extinct)),
#             html.Td(html.P(eta if time_of_finishing is None else "       ")),
#             html.Td(html.P(str(basepath))),
#             html.Td(
#                 html.Button(
#                     "delete",
#                     id={
#                         "type": "delete-simulation-button",
#                         "index": filename,
#                     },
#                     value=filename,
#                 ),
#                 style={"padding-right": "1rem"},
#             ),
#         ],
#     )
#     return row


# # CREATE
# @callback(
#     Output("list-section-table", "children"),
#     Input("list-view-button", "n_clicks"),
#     State({"type": "selection-state", "index": ALL}, "data"),
#     State("list-section-table", "children"),
#     # Input({"type": "delete-simulation-button", "index": ALL}, "children"),
#     # prevent_initial_call=True,
# )
# @funcs.print_function_name
# def refresh_list_section(n_clicks, data, children):
#     if children is not None:
#         print(data)
#         return children

#     # Get data from folders
#     paths = funcs.get_sim_paths()
#     containers = [Container(path) for path in paths]

#     # Make table
#     # - specify table header
#     table_header = html.Tr(
#         [
#             html.Th("ID", style={"padding-left": "1.3rem", "padding-right": "0.8rem"}),
#             html.Th("DISPLAY"),
#             html.Th("CREATED"),
#             html.Th("FINISHED"),
#             html.Th("EXTINCT"),
#             html.Th("ETA"),
#             html.Th("FILEPATH"),
#             html.Th("DELETE", style={"padding-right": "1rem"}),
#         ],
#     )
#     table = [table_header]

#     # - add table rows
#     for container in containers:
#         element = make_table_row(container)
#         table.append(element)

#     return html.Table(children=table, className="list-table")
