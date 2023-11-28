def run():
    from dash import Dash, html, dcc, callback, Output, Input, State
    from aegis.visor.layout import app_layout
    import aegis.visor.callbacks
    import aegis.visor.tab_config.callbacks_config
    import aegis.visor.tab_plot.callbacks_plot
    import aegis.visor.tab_list.callbacks_list

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title=None,
    )
    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout
    app.run(debug=True)
