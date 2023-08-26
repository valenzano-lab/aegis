# Import packages
from dash import Dash, html, dcc, callback, Output, Input, State
from aegis.visor.layout import app_layout
import aegis.visor.callbacks
import aegis.visor.callbacks_plot


app = Dash(__name__, suppress_callback_exceptions=True)
app._favicon = "favicon.ico"
app.title = "AEGIS visualizer"
app.layout = app_layout
if __name__ == "__main__":
    app.run(debug=True)
