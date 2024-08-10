import dash
import dash_bootstrap_components as dbc
from aegis.utilities.container import Container
from aegis_gui.utilities import utilities
from aegis.modules.initialization.parameterization.default_parameters import DEFAULT_PARAMETERS


def make_select(selected=None):

    paths = utilities.get_sim_paths()

    if not paths:
        return []

    if selected:
        for path in paths:
            if Container(path).name == selected:
                selected_path = path
                break
    else:
        selected_path = paths[0]

    return [
        dash.html.P("Copy parameters"),
        dbc.Select(
            id="sim-config-select",
            options=[{"label": path.stem, "value": str(path)} for path in paths],
            # value=["default"] if "default" in dropdown_options else [],
            value=paths[0].stem,
            placeholder="Copy simulation",
            # className="plot-dropdown",
            # multiple=True,
            style={"margin-bottom": "1rem"},
            className="me-2",
        ),
        dbc.Button(
            [dash.html.I(className="bi bi-arrow-up-square-fill"), "copy"],
            id="sim-config-copy",
            className="me-1",
            # outline=True,
            color="primary",
        ),
        dbc.Button(
            [dash.html.I(className="bi bi-x-circle-fill"), "reset"],
            id="reset-run-button",
            className="me-1",
            # outline=True,
            color="danger",
        ),
    ]


@dash.callback(
    dash.Output({"type": "config-input", "index": dash.ALL}, "value", allow_duplicate=True),
    dash.Input("sim-config-copy", "n_clicks"),
    dash.State("sim-config-select", "value"),
    dash.State({"type": "config-input", "index": dash.ALL}, "id"),
    dash.State({"type": "config-input", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def reset_configs(n_clicks, filename, ids, current_values):
    """Why not use DEFAULT_PARAMETERS.get_default_parameters()? To ensure that the order of output values is correct."""

    if n_clicks is None:
        return dash.no_update

    container = utilities.get_container(filename=filename)
    config = container.get_config()

    new_values = []

    for id_, current_value in zip(ids, current_values):
        param_name = id_["index"]
        new_value = config[param_name]
        # new_value = param.default
        if new_value == current_value:
            new_values.append(dash.no_update)
        else:
            new_values.append(new_value)

    return new_values
