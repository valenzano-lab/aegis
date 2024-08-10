import os
import subprocess
import dash
import pathlib
import dash_bootstrap_components as dbc


def make_button(path):
    return dbc.Button(
        children=[dash.html.I(className="bi bi-folder-symlink-fill"), "Open data location"],
        id={"type": "config-basepath-folder", "index": str(path)},
        value=str(path),
        className="me-2",
    )


@dash.callback(
    dash.Output({"type": "config-basepath-folder", "index": dash.MATCH}, "n_clicks"),
    [dash.Input({"type": "config-basepath-folder", "index": dash.MATCH}, "n_clicks")],
    prevent_initial_call=True,
)
def open_file_manager(n_clicks):

    path = pathlib.Path(dash.ctx.triggered_id["index"])

    if n_clicks > 0:
        # For Windows
        if os.name == "nt":
            os.startfile(path)
        # For macOS
        elif os.name == "posix":
            if "darwin" in os.uname().sysname.lower():  # macOS check
                subprocess.run(["open", path])
            else:  # Assume Linux or WSL
                # Check if running under WSL
                if "microsoft" in os.uname().release.lower():
                    # Convert the WSL path to a Windows path and use explorer.exe
                    windows_path = subprocess.check_output(["wslpath", "-w", path]).decode().strip()
                    subprocess.run(["explorer.exe", windows_path])
                else:  # Regular Linux
                    subprocess.run(["xdg-open", path])
    return 0  # Reset the click count
