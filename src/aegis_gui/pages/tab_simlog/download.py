from dash import html, dcc, Output, MATCH, Input, State, callback
import yaml
from aegis_gui.utilities import log_funcs
from aegis.utilities.container import Container


def make_button(filename):
    return html.Button(
        "download",
        id={"type": "config-download-button", "index": filename},
        value=filename,
    )


def make_dcc(filename):
    return dcc.Download(id={"type": "config-dcc-download", "index": filename})


@callback(
    Output({"type": "config-dcc-download", "index": MATCH}, "data"),
    Input({"type": "config-download-button", "index": MATCH}, "n_clicks"),
    State({"type": "config-download-basepath", "index": MATCH}, "children"),
)
@log_funcs.log_debug
def config_file_download_button(n_clicks, basepath):
    if n_clicks is None:
        return n_clicks

    container = Container(basepath)
    config = container.get_config()
    return {
        "content": yaml.dump(config),
        "filename": container.name + ".yml",
    }
