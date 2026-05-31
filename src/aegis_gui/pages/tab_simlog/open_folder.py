import os
import logging
import subprocess
import dash
import pathlib
import dash_bootstrap_components as dbc
import platform

from aegis_gui.guisettings.GuiSettings import gui_settings
from aegis_gui.utilities import utilities


def make_button(path):
    # Index is the sim name (path.name / path.stem), not the absolute path.
    # The callback resolves the path server-side; never trust client-supplied
    # paths for filesystem operations.
    name = pathlib.Path(path).name
    return dbc.Button(
        children=[dash.html.I(className="bi bi-folder-symlink-fill"), "Open data location"],
        id={"type": "config-basepath-folder", "index": name},
        value=name,
        color="secondary",
        className="me-2",
    )


@dash.callback(
    dash.Output({"type": "config-basepath-folder", "index": dash.MATCH}, "n_clicks"),
    [dash.Input({"type": "config-basepath-folder", "index": dash.MATCH}, "n_clicks")],
    prevent_initial_call=True,
)
def open_file_manager(n_clicks):
    """Open the file manager at the specified path in a platform-independent way.

    Hard-gated to local environments. This callback spawns a desktop-app
    subprocess, which makes zero sense on a remote server — and would be a
    serious risk if exposed (subprocess on a server-trusted path).
    """
    # Server-side environment guard (in addition to the render-time guard in
    # dropdown.py that hides the button). Defense in depth.
    if gui_settings.ENVIRONMENT != "local":
        logging.warning("open_file_manager called in non-local environment; ignored.")
        return 0

    triggered = dash.ctx.triggered_id
    if not isinstance(triggered, dict) or "index" not in triggered:
        return 0

    try:
        path = utilities.safe_sim_path(triggered["index"])
    except utilities.UnsafeSimNameError as exc:
        logging.warning("Rejected open_file_manager request: %s", exc)
        return 0

    # TODO does not work on WSL

    if n_clicks > 0:
        system = platform.system().lower()

        if system == "windows":
            os.startfile(path)  # Opens the file manager on Windows
        elif system == "darwin":
            subprocess.run(["open", path])  # Opens the file manager on macOS
        elif system == "linux":
            subprocess.run(["xdg-open", path])  # Opens the file manager on Linux
        elif "microsoft" in os.uname().release.lower():  # For WSL (Windows Subsystem for Linux)
            windows_path = subprocess.check_output(["wslpath", "-w", path]).decode().strip()
            subprocess.run(["explorer.exe", windows_path])  # Opens the Windows Explorer

    return 0  # Reset the click count
