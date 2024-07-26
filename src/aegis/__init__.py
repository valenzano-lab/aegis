"""
This script is executed when AEGIS is imported (`import aegis`). Execute functions by running `aegis.run_from_{}`.
AEGIS can be started in multiple ways; each of these functions starts AEGIS from a different context. 
"""

import logging

from aegis.manager import Manager
from aegis.modules.initialization.terminalmanager import parse_terminal


def run_from_script(custom_config_path, pickle_path, overwrite, custom_input_params):
    """
    Use this function when you want to run aegis from a python script.

    $ import aegis
    $ aegis.run()
    """

    manager = Manager(
        custom_config_path=custom_config_path,
        pickle_path=pickle_path,
        overwrite=overwrite,
        custom_input_params=custom_input_params,
    )
    manager.run()


def run_from_main():
    run_from_terminal()


def run_from_terminal():
    custom_config_path, pickle_path, overwrite, server_mode = parse_terminal()

    if (custom_config_path, pickle_path, overwrite) == (None, None, None):
        logging.info("Starting run_gui.")
        logging.info(f"Server mode is {'ON' if server_mode else 'OFF'}.")
        if server_mode:
            run_from_server_gui()
        else:
            run_from_local_gui()
    else:
        manager = Manager(
            custom_config_path=custom_config_path,
            pickle_path=pickle_path,
            overwrite=overwrite,
            custom_input_params={},
        )
        manager.run()


def run_from_server_gui():
    import aegis_gui

    aegis_gui.run(environment="server")


def run_from_local_gui():
    import aegis_gui

    aegis_gui.run(environment="local")
