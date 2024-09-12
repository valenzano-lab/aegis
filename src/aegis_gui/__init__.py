from aegis_gui import app
from aegis_gui.guisettings.GuiSettings import gui_settings


def run(environment):
    gui_settings.set(environment=environment)
    _app = app.get_app()
    _app.run(debug=gui_settings.DEBUG_MODE)
