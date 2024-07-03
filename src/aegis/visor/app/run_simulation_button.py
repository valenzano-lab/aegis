from dash import html, dcc, callback, Output, Input, State, ALL
from aegis.visor import utilities
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS

layout = html.Div(
    id="sim-section-control",
    children=[
        dcc.Input(
            id="config-make-text",
            className="control-element",
            type="text",
            placeholder="unique id",
            autoComplete="off",
        ),
        html.Button(
            "run simulation",
            id="simulation-run-button",
            className="control-element",
        ),
        html.P("", id="simulation-run-text"),
        # html.Button("make config", id="config-make-button"),]
    ],
)


def decode_config_tab_values(values, ids_):
    for value, id_ in zip(values, ids_):
        param = DEFAULT_PARAMETERS[id_["index"]]
        if param.dtype == bool:
            yield id_["index"], bool(value)
        else:
            yield id_["index"], value


@callback(
    Output("config-make-text", "value"),
    Input("simulation-run-button", "n_clicks"),
    State("config-make-text", "value"),
    State({"type": "config-input", "index": ALL}, "value"),
    State({"type": "config-input", "index": ALL}, "id"),
    prevent_initial_call=True,
)
@utilities.log_debug
def click_sim_button(n_clicks, filename, values, ids_):
    """
    Run simulation when sim button clicked (also, implicitly, not disabled).
    """
    if n_clicks is None:
        return

    # make config file

    decoded_pairs = list(decode_config_tab_values(values=values, ids_=ids_))
    input_config = {i: v for i, v in decoded_pairs}
    utilities.make_config_file(filename, input_config)

    # run simulation
    utilities.run(filename)

    return ""
