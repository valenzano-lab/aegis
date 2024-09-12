import dash
import dash_bootstrap_components as dbc
from aegis_gui.guisettings.GuiSettings import gui_settings
from aegis_gui.utilities import manipulate, utilities
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from .valid_range import is_input_in_valid_range


layout = dash.html.Div(
    [
        dash.html.H6("Run simulation"),
        dash.html.Div(
            id="sim-section-control",
            children=[
                dash.html.Div(
                    children=[
                        dbc.InputGroup(
                            children=[
                                dbc.InputGroupText("Simulation ID"),
                                dbc.Input(
                                    id="config-make-text",
                                    type="text",
                                    placeholder="Enter a unique ID",
                                    # autoComplete="off",
                                    className="me-2",
                                ),
                            ],
                        ),
                        dbc.FormText("", id="config-make-error-text", style={"marginLeft": "8.3rem"}),
                    ],
                    id="config-make-text-group",
                ),
                dbc.Button(
                    [dash.html.I(className="bi bi-rocket-takeoff-fill"), "Launch"],
                    id="simulation-run-button",
                    className="me-1",
                    outline=True,
                    color="success",
                ),
                # dash.html.Button("make config", id="config-make-button"),]
                # dbc.FormFeedback(
                #     "Enter a unique simulation ID",
                #     type="invalid",
                # ),
                # dash.html.P("sssssssssssss", id="simulation-run-text"),
            ],
            style={"display": "flex", "align-items": "flex-start"},
        ),
    ]
)


def decode_config_tab_values(values, ids_):
    for value, id_ in zip(values, ids_):
        param = DEFAULT_PARAMETERS[id_["index"]]
        if param.dtype == bool:
            assert value in {
                "True",
                "False",
                True,
                False,
            }, f"{param.key} value is '{value}' but it should be True or False"
            yield id_["index"], value in {"True", True}
        else:
            yield id_["index"], value


@dash.callback(
    dash.Output({"type": "config-input", "index": dash.ALL}, "value", allow_duplicate=True),
    dash.Input("reset-run-button", "n_clicks"),
    dash.State({"type": "config-input", "index": dash.ALL}, "id"),
    dash.State({"type": "config-input", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def reset_configs(n_clicks, ids, current_values):
    """Why not use DEFAULT_PARAMETERS.get_default_parameters()? To ensure that the order of output values is correct."""

    if n_clicks is None:
        return

    new_values = []

    for id_, current_value in zip(ids, current_values):
        param_name = id_["index"]
        param = DEFAULT_PARAMETERS[param_name]
        new_value = param.default
        if new_value == current_value:
            new_values.append(dash.no_update)
        else:
            new_values.append(new_value)

    return new_values


@dash.callback(
    dash.Output("config-make-text", "value"),
    dash.Input("simulation-run-button", "n_clicks"),
    dash.State("config-make-text", "value"),
    dash.State({"type": "config-input", "index": dash.ALL}, "value"),
    dash.State({"type": "config-input", "index": dash.ALL}, "id"),
    dash.State("prerun-sim-select", "value"),
    prevent_initial_call=True,
)
def click_sim_button(n_clicks, filename, values, ids_, prerun_sim_path):
    """
    Run simulation when sim button clicked (also, implicitly, not disabled).
    """
    if n_clicks is None:
        return

    # make config file
    decoded_pairs = list(decode_config_tab_values(values=values, ids_=ids_))
    input_config = {i: v for i, v in decoded_pairs}
    manipulate.make_config_file(filename, input_config)

    # run simulation
    manipulate.run_simulation(filename, prerun_sim_path=prerun_sim_path)
    return ""


# def is_sim_name_valid(sim_name: str) -> bool:
#     return (
#         (sim_name is not None) and (sim_name != "") and ("." not in sim_name) and (len(sim_name) < MAX_SIM_NAME_LENGTH)
#     )


@dash.callback(
    dash.Output("simulation-run-button", "outline"),
    dash.Output("simulation-run-button", "disabled"),
    dash.Output("config-make-text", "invalid"),
    dash.Output("config-make-text", "valid"),
    dash.Output("config-make-error-text", "children"),
    dash.Input("config-make-text", "value"),
    dash.Input({"type": "config-input", "index": dash.ALL}, "value"),
    dash.State({"type": "config-input", "index": dash.ALL}, "id"),
    dash.State("ticker-store", "data"),
)
def disable_sim_button(filename, values, ids, ticker_store) -> bool:
    """
    Make simulation run button unclickable when any of these is true:
    - input values for parameters are not inside the acceptable server range
    - simulation ID is not valid
    - simulation ID is already used
    """

    block = (True, True, True, False)

    # TODO block here is not correct because the simname is fine
    for value, id_ in zip(values, ids):
        param_name = id_["index"]
        is_value_set = value != "" and value is not None
        if not is_value_set:
            continue

        # check validity of input value
        in_valid_range = is_input_in_valid_range(input_=value, param_name=param_name)
        if not in_valid_range:
            return (
                *block,
                f"Parameter {param_name} is out of valid range",
            )

    # TODO do not use block bc that is not the problem
    currently_running = sum(ticker_store.values())
    if not gui_settings.can_run_more_simulations(currently_running=currently_running):
        return *block, f"Maximum number of simulations running is reached ({currently_running})"

    if filename is None or filename == "":
        return True, True, False, False, ""

    if "." in filename:
        return *block, "ID cannot contain a period (.)"

    if len(filename) > gui_settings.MAX_SIM_NAME_LENGTH:
        return *block, f"ID has more than {gui_settings.MAX_SIM_NAME_LENGTH} characters"

    if utilities.sim_exists(filename):
        return *block, "Entered ID has already been used"

    return False, False, False, True, ""
