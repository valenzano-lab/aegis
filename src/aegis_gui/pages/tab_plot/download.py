import logging
from dash import callback, Output, Input, ctx, State, MATCH, dcc, html
from plotly.io import write_image
from aegis_gui.utilities.utilities import get_figure_dir


@callback(
    Output({"type": "figure-dcc-download", "index": MATCH}, "data"),
    Input({"type": "figure-download-button", "index": MATCH}, "n_clicks"),
    State({"type": "graph-figure", "index": MATCH}, "figure"),
)
def figure_download_button_click(n_clicks, figure):

    if n_clicks is None:
        return

    fig_name = ctx.triggered_id["index"]
    path_figure = get_figure_dir() / f"{fig_name}.png"
    logging.info(f"Recording figure to {path_figure}...")
    write_image(figure, path_figure)
    logging.info(f"'{fig_name}' has been recorded successfully.")

    return dcc.send_file(path_figure)


def get_figure_download_button(figure_id):
    return html.Button("download figure", id={"type": "figure-download-button", "index": figure_id})


def get_figure_download_dcc(figure_id):
    return dcc.Download(id={"type": "figure-dcc-download", "index": figure_id})
