import os
import subprocess
import dash
import pathlib
import dash_bootstrap_components as dbc
import platform


def make_button(path):
    return dbc.Button(
        children=[dash.html.I(className="bi bi-folder-symlink-fill"), "Open data location"],
        id={"type": "config-basepath-folder", "index": str(path)},
        value=str(path),
        color="dark",
        className="me-2",
    )


@dash.callback(
    dash.Output({"type": "config-basepath-folder", "index": dash.MATCH}, "n_clicks"),
    [dash.Input({"type": "config-basepath-folder", "index": dash.MATCH}, "n_clicks")],
    prevent_initial_call=True,
)
def open_file_manager(n_clicks):
    """Open the file manager at the specified path in a platform-independent way."""
    path = pathlib.Path(dash.ctx.triggered_id["index"])

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
