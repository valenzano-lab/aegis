import dash
from dash import callback, Output, Input, State, MATCH, no_update, dcc, ctx
from aegis_gui.utilities import log_funcs
from .valid_range import is_input_in_valid_range
import dash_bootstrap_components as dbc
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS

# @callback(
#     Output({"type": "config-input", "index": dash.ALL}, "invalid"),
#     Input({"type": "info-badge", "index": dash.ALL}, "children"),
#     Input({"type": "info-badge", "index": dash.ALL}, "children"),
# )
# @log_funcs.log_debug
# def handle_config_input(value) -> bool:
#     """
#     Change style of config input so that the user knows that the input value is outside of valid server range.
#     """
#     if ctx.triggered_id is None:
#         return dash.no_update, dash.no_update

#     param_name = ctx.triggered_id["index"]
#     valid = is_input_in_valid_range(input_=value, param_name=param_name)

#     parameter = DEFAULT_PARAMETERS[param_name]
#     value_is_default = value == parameter.default

#     return not valid, "" if value_is_default else "modified"


@callback(
    Output({"type": "collapse-badge", "index": dash.ALL}, "children"),
    Output({"type": "collapse-badge", "index": dash.ALL}, "color"),
    Input({"type": "info-badge", "index": dash.ALL}, "children"),
    State({"type": "info-badge", "index": dash.ALL}, "id"),
    State({"type": "collapse-badge", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def update_badge(badges, id_params, id_domains):
    pnames = [idp["index"] for idp in id_params]
    domains = [idd["index"] for idd in id_domains]
    statuses = {domain: 0 for domain in domains}

    for badge, pname in zip(badges, pnames):
        if badge == "modified":
            domain = DEFAULT_PARAMETERS[pname].domain
            statuses[domain] += 1

    statuses = [statuses[domain] for domain in domains]
    colors = ["primary" if status > 0 else "secondary" for status in statuses]

    return statuses, colors


@callback(
    Output({"type": "config-input", "index": MATCH}, "invalid"),
    Output({"type": "info-badge", "index": MATCH}, "children"),
    Input({"type": "config-input", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def handle_config_input(value) -> bool:
    """
    Change style of config input so that the user knows that the input value is outside of valid server range.
    """
    if ctx.triggered_id is None:
        return dash.no_update, dash.no_update

    param_name = ctx.triggered_id["index"]
    valid = is_input_in_valid_range(input_=value, param_name=param_name)

    parameter = DEFAULT_PARAMETERS[param_name]
    value_is_default = value == parameter.default

    return not valid, "" if value_is_default else "modified"


def get_input_element(param):
    """
    Generate appropriate Dash input component based on the parameter's data type.
    """
    common_props = {"id": {"type": "config-input", "index": param.key}, "className": "config-input-class"}

    if param.dtype in [int, float]:
        step = 0 if param.dtype == int else 0.01
        return dbc.Input(
            type="number",
            value=param.default,
            step=step,
            autoComplete="off",
            **common_props,
        )

    if param.dtype == bool:
        options = ["True", "False"]
        return dbc.Select(options=[{"label": k, "value": k} for k in options], value=str(param.default), **common_props)

    if param.dtype == str:
        options = param.drange.strip("{}").split(", ")
        return dbc.Select(options=[{"label": k, "value": k} for k in options], value=param.default, **common_props)

    # TODO: resolve dict parameter
    return
