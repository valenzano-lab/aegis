import pathlib
from dash import html
from aegis.utilities.get_folder_size import get_folder_size_with_du
from aegis.visor.utilities import log_funcs


@log_funcs.log_debug
def get_simlog_layout(path: pathlib.Path):
    folder_size = get_folder_size_with_du(path)
    return html.Td(html.P(folder_size))
