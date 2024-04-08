import logging
from dash import callback, Output, Input, ctx, State, MATCH
from plotly.io import write_image
from visor.utilities import get_figure_dir


@callback(
    Output({"type": "figure-download-button", "index": MATCH}, "n_clicks"),
    Input({"type": "figure-download-button", "index": MATCH}, "n_clicks"),
    State({"type": "graph-figure", "index": MATCH}, "figure"),
)
def create_figure(n_clicks, figure):

    if n_clicks is None:
        return n_clicks

    fig_name = ctx.triggered_id["index"]
    path_figure = get_figure_dir() / f"{fig_name}.png"
    logging.info(f"Recording '{fig_name}'...")
    write_image(figure, path_figure)
    logging.info(f"'{fig_name}' has been recorded successfully.")

    return n_clicks
