from dash import callback, Output, Input, State, ALL, MATCH, ctx
from aegis.visor.utilities import log_funcs
from .valid_range import is_input_in_valid_range


@callback(
    Output({"type": "config-input", "index": MATCH}, "className"),
    Input({"type": "config-input", "index": MATCH}, "value"),
    State({"type": "config-input", "index": MATCH}, "className"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def disable_config_input(value, className) -> str:
    """
    Change style of config input so that the user knows that the input value is outside of valid server range.
    """
    if ctx.triggered_id is None:
        return className

    param_name = ctx.triggered_id["index"]
    className = className.replace(" disabled", "")

    in_valid_range = is_input_in_valid_range(input_=value, param_name=param_name)
    if not in_valid_range:
        className += " disabled"

    return className
