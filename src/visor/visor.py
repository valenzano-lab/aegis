from dash import Dash
from visor.layout import app_layout
import visor.callbacks
import visor.tab_config.callbacks_config
import visor.tab_plot.callbacks_plot
import visor.tab_list.callbacks_list


def run():

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title="",
    )
    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout
    app.run(debug=True)
