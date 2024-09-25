import dash
from dash import callback, Output, Input, State, MATCH, no_update, dcc, ctx
from aegis_gui.utilities import log
from .valid_range import is_input_in_valid_range
import dash_bootstrap_components as dbc
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS
from aegis_sim.parameterization.parameter import Parameter

# @callback(
#     Output({"type": "config-input", "index": dash.ALL}, "invalid"),
#     Input({"type": "info-badge", "index": dash.ALL}, "children"),
#     Input({"type": "info-badge", "index": dash.ALL}, "children"),
# )
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

#     return not valid, "" if value_is_default else "reset"


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
        if badge == "reset":
            domain = DEFAULT_PARAMETERS[pname].domain
            statuses[domain] += 1

    texts = [f"{statuses[domain]} modifications" for domain in domains]
    colors = ["danger" if statuses[domain] > 0 else "secondary" for domain in domains]

    return texts, colors


@callback(
    Output({"type": "config-input", "index": MATCH}, "invalid"),
    Output({"type": "info-badge", "index": MATCH}, "children"),
    Output({"type": "info-badge", "index": MATCH}, "color"),
    Input({"type": "config-input", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def handle_config_input(value) -> bool:
    """
    Change style of config input so that the user knows that the input value is outside of valid server range.
    """
    if ctx.triggered_id is None:
        return dash.no_update, dash.no_update, dash.no_update

    param_name = ctx.triggered_id["index"]
    valid = is_input_in_valid_range(input_=value, param_name=param_name)

    parameter = DEFAULT_PARAMETERS[param_name]
    value_is_default = value == parameter.default
    new_value = "default" if value_is_default else "reset"
    new_color = "danger" if new_value == "reset" else "secondary"

    return not valid, new_value, new_color


def get_input_element(param: Parameter):
    """
    Generate appropriate Dash input component based on the parameter's data type.
    """

    if not param.show_in_gui:
        return

    common_props = {"id": {"type": "config-input", "index": param.key}, "className": "config-input-class"}

    if param.dtype in [int, float]:
        step = 0 if param.dtype == int else 0.01

        input_element = dbc.Input(
            type="number",
            value=param.default,
            step=step,
            autoComplete="off",
            **common_props,
        )

    elif param.dtype == bool:
        options = ["True", "False"]
        input_element = dbc.Select(
            options=[{"label": k, "value": k} for k in options], value=str(param.default), **common_props
        )

    elif param.dtype == str:
        options = param.drange.strip("{}").split(", ")
        input_element = dbc.Select(
            options=[{"label": k, "value": k} for k in options], value=param.default, **common_props
        )

    else:
        # TODO: resolve dict parameter
        return

    return dbc.InputGroup(
        children=[
            dbc.InputGroupText(
                children=[
                    dash.html.Span(param.get_name(), className="tooltip-underline" if param.info else ""),
                    (
                        dbc.Tooltip(
                            param.info,
                            target={"type": "info-tooltip", "index": param.key},
                            placement="right",
                        )
                        if param.info
                        else None
                    ),
                ],
                id={"type": "info-tooltip", "index": param.key},
            ),
            input_element,
            dbc.InputGroupText(
                [
                    dbc.Badge(
                        "default",
                        pill=True,
                        id={"type": "info-badge", "index": param.key},
                        color="secondary",
                        style={"width": "80px", "cursor": "pointer"},
                    )
                ],
                id={"type": "input-modified-state", "index": param.key},
            ),
        ],
        className="mb-2 mt-2",
    )


@dash.callback(
    Output({"type": "info-badge", "index": dash.MATCH}, "children", allow_duplicate=True),
    Output({"type": "config-input", "index": dash.MATCH}, "value"),
    Input({"type": "info-badge", "index": dash.MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def reset_config_input(n_clicks):

    if n_clicks is None:
        return dash.no_update, dash.no_update

    param_name = ctx.triggered_id["index"]
    new_value = DEFAULT_PARAMETERS[param_name].default
    return "default", new_value
