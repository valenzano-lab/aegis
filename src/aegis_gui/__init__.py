from .app import get_app


def run(environment):
    from . import config

    config.set(environment=environment)
    app = get_app()
    app.run(debug=config.config.debug_mode)
