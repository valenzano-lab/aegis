def run():
    from dash import Dash, html, dcc, callback, Output, Input, State
    from visor.layout import app_layout
    import visor.callbacks
    import visor.tab_config.callbacks_config
    import visor.tab_plot.callbacks_plot
    import visor.tab_list.callbacks_list

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title=None,
    )
    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout
    app.run(debug=True)
