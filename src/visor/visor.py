"""
Import callbacks. They modify Output's when Input's are triggered. Callbacks add reactivity to the app.
"""

from dash import Dash
from visor.app.layout import app_layout
import visor.app.callbacks
import visor.tab_config.callbacks_config
import visor.tab_plot.callbacks_plot
import visor.tab_list.callbacks_list


def run():

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title="",
        # *.css in assets are automatically imported; they need to be explicitly ignored
        assets_ignore="styles-dark.css",
    )
    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout
    app.run(debug=True)
