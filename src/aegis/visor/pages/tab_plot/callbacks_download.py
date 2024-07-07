import logging
from dash import callback, Output, Input, ctx, State, MATCH, dcc
from plotly.io import write_image
from aegis.visor.utilities.utilities import get_figure_dir


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
