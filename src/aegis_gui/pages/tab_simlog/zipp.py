from dash import html, dcc, Output, Input, MATCH, callback, ctx, no_update
import logging
import pathlib
import zipfile
import io
from aegis_gui.utilities import log, utilities
import dash_bootstrap_components as dbc


def get_zip_button_layout(filename):
    return html.Div(
        children=[
            dbc.Button(
                [html.I(className="bi bi-file-earmark-zip-fill"), "Download data"],
                id={"type": "zip-download-button", "index": filename},
                value=filename,
                className="me-2",
                color="secondary",
            ),
            dcc.Download(id={"type": "zip-dcc-download", "index": filename}),
        ],
    )


@callback(
    Output({"type": "zip-dcc-download", "index": MATCH}, "data"),
    Input({"type": "zip-download-button", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True,
    # running=[(Output("zip-download-button", "disabled"), True, False)] # currently not supported
)
def generate_zip(n_clicks):
    if n_clicks is None:
        return no_update
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or "index" not in triggered:
        return no_update
    try:
        # Derive the path server-side from the button's index (the sim name).
        # NEVER trust an absolute path sent by the browser.
        folder_path = utilities.safe_sim_path(triggered["index"])
    except utilities.UnsafeSimNameError as exc:
        logging.warning("Rejected zip request: %s", exc)
        return no_update
    zip_buffer = zip_folder(folder_path)
    return dcc.send_bytes(zip_buffer.getvalue(), f"{folder_path.name}.zip")


# Zipping function


def zip_folder(folder_path: pathlib.Path) -> io.BytesIO:
    if not folder_path.is_dir():
        raise ValueError("The provided path is not a directory")

    # Create a BytesIO object to hold the zipped data
    zip_buffer = io.BytesIO()

    # Create a Zip archive in memory
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in folder_path.rglob("*"):
            zip_file.write(file, file.relative_to(folder_path))

    # Seek to the beginning of the BytesIO object
    zip_buffer.seek(0)
    return zip_buffer
