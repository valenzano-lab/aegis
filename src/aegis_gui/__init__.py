from aegis_gui import app
from aegis_gui.guisettings.GuiSettings import gui_settings


def run(environment, debug):
    gui_settings.set(environment=environment, debug=debug)
    _app = app.get_app()
    if gui_settings.PORT is None:
        _app.run(debug=gui_settings.DEBUG_MODE)
    else:
        _app.run(debug=gui_settings.DEBUG_MODE, port=gui_settings.PORT)
