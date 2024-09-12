from aegis_gui import app
from aegis_gui.config import config


def run(environment):
    config.set(environment=environment)
    _app = app.get_app()
    _app.run(debug=config.config.DEBUG_MODE)
