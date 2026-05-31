from dash import html, dcc, Output, MATCH, Input, callback, ctx, no_update
import logging
import yaml
from aegis_gui.utilities import log, utilities
from aegis_sim.utilities.container import Container

import dash_bootstrap_components as dbc


def make_button(filename):
    return dbc.Button(
        [html.I(className="bi bi-gear-fill"), "Download configuration"],
        id={"type": "config-download-button", "index": filename},
        value=filename,
        color="secondary",
        className="me-2",
    )


def make_dcc(filename):
    return dcc.Download(id={"type": "config-dcc-download", "index": filename})


@callback(
    Output({"type": "config-dcc-download", "index": MATCH}, "data"),
    Input({"type": "config-download-button", "index": MATCH}, "n_clicks"),
)
def config_file_download_button(n_clicks):
    if n_clicks is None:
        return no_update
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or "index" not in triggered:
        return no_update
    try:
        # Resolve the path server-side from the button's index (the sim name);
        # do not trust a browser-supplied basepath.
        sim_path = utilities.safe_sim_path(triggered["index"])
    except utilities.UnsafeSimNameError as exc:
        logging.warning("Rejected config download: %s", exc)
        return no_update

    container = Container(sim_path)
    config = container.get_config()
    return {
        "content": yaml.dump(config),
        "filename": container.name + ".yml",
    }
