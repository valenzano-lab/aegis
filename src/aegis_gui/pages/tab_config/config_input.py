from dash import callback, Output, Input, State, MATCH, no_update, dcc, ctx
from aegis_gui.utilities import log_funcs
from .valid_range import is_input_in_valid_range
import dash_bootstrap_components as dbc


@callback(
    Output({"type": "config-input", "index": MATCH}, "invalid"),
    Input({"type": "config-input", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def disable_config_input(value) -> bool:
    """
    Change style of config input so that the user knows that the input value is outside of valid server range.
    """
    if ctx.triggered_id is None:
        return no_update

    param_name = ctx.triggered_id["index"]
    valid = is_input_in_valid_range(input_=value, param_name=param_name)
    return not valid


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
