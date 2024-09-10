import dash
import dash_bootstrap_components as dbc
from aegis_gui import layout


@dash.callback(
    [dash.Output(f"link-nav-{page}", "active") for page in ["home", "config", "plot", "simlog", "wiki"]],
    [dash.Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname is None:
        # Default to home if pathname is None
        pathname = "/"
    return [
        pathname == f"/{page}" or (page == "home" and pathname == "/")
        for page in ["home", "config", "plot", "simlog", "wiki"]
    ]


def get_app():
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title="",
        # *.css in assets are automatically imported; they need to be explicitly ignored
        assets_ignore="styles-dark.css",
        use_pages=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    )

    # Bootstrap ICONS: https://icons.getbootstrap.com/

    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = layout.get_app_layout()

    return app
