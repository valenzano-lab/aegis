from aegis_gui import config, app


def run(environment):
    config.set(environment=environment)
    _app = app.get_app()
    _app.run(debug=config.config.debug_mode)
