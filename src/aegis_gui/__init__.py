from dash import Dash
import dash_bootstrap_components as dbc


def run(environment):
    from . import config

    config.set(environment=environment)

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title="",
        # *.css in assets are automatically imported; they need to be explicitly ignored
        assets_ignore="styles-dark.css",
        use_pages=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )

    from aegis_gui.app.layout import app_layout

    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout
    app.run(debug=config.config.debug_mode)
