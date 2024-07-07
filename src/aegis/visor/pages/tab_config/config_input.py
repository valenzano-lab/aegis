from dash import callback, Output, Input, State, MATCH, no_update, dcc, ctx
from aegis.visor.utilities import log_funcs
from .valid_range import is_input_in_valid_range


@callback(
    Output({"type": "config-input", "index": MATCH}, "className"),
    Input({"type": "config-input", "index": MATCH}, "value"),
    State({"type": "config-input", "index": MATCH}, "className"),
    prevent_initial_call=True,
)
@log_funcs.log_debug
def disable_config_input(value, class_name) -> str:
    """
    Change style of config input so that the user knows that the input value is outside of valid server range.
    """
    if ctx.triggered_id is None:
        return no_update

    param_name = ctx.triggered_id["index"]
    class_names = set(class_name.split())

    # Remove 'disabled' class if it exists
    class_names.discard("disabled")

    # Check if the input value is within the valid range
    if not is_input_in_valid_range(input_=value, param_name=param_name):
        class_names.add("disabled")

    # Return the updated class names as a space-separated string
    updated_class_name = " ".join(class_names)
    
    return updated_class_name


def get_input_element(param):
    """
    Generate appropriate Dash input component based on the parameter's data type.
    """
    common_props = {"id": {"type": "config-input", "index": param.key}, "className": "config-input-class"}

    if param.dtype in [int, float]:
        step = 0 if param.dtype == int else 0.01
        return dcc.Input(
            type="number",
            value=param.default,
            step=step,
            autoComplete="off",
            **common_props,
        )

    if param.dtype == bool:
        return dcc.Checklist(
            [""],
            value=[""] if param.default else [],
            **common_props,
        )

    if param.dtype == str:
        options = param.drange.strip("{}").split(", ")
        return dcc.Dropdown(
            options=options,
            value=param.default,
            searchable=False,
            clearable=False,
            **common_props,
        )

    # TODO: resolve dict parameter
    return
