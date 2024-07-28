import logging
import dash_bootstrap_components as dbc
import dash
import pandas as pd
from dash import callback, Output, Input, ctx, State, MATCH, dcc, html
from plotly.io import write_image
from aegis_gui.utilities.utilities import get_figure_dir

# FIGURE DOWNLOAD


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
    button = dbc.Button(
        [html.I(className="bi bi-file-arrow-down-fill"), "Download figure"],
        id={"type": "figure-download-button", "index": figure_id},
    )
    return button


def get_figure_download_dcc(figure_id):
    return dcc.Download(id={"type": "figure-dcc-download", "index": figure_id})


# DATA DOWNLOAD


@callback(
    Output({"type": "data-dcc-download", "index": MATCH}, "data"),
    Input({"type": "data-download-button", "index": MATCH}, "n_clicks"),
)
def download_data(n_clicks):
    if n_clicks is None:
        return
    # Convert the sample data to a DataFrame
    df = pd.DataFrame([1, 2, 3, 4])

    # Convert DataFrame to CSV
    csv_string = df.to_csv(index=False)

    return dcc.send_data_frame(df.to_csv, "data.csv")


def get_data_download_button(figure_id):
    button = dbc.Button(
        [html.I(className="bi bi-file-arrow-down-fill"), "Download data"],
        id={"type": "data-download-button", "index": figure_id},
    )
    return button


def get_data_download_dcc(figure_id):
    return dcc.Download(id={"type": "data-dcc-download", "index": figure_id})
