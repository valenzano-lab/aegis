"""
This script is executed when AEGIS is imported (`import aegis`). Execute functions by running `aegis.run_from_{}`.
AEGIS can be started in multiple ways; each of these functions starts AEGIS from a different context. 
"""

import logging

from aegis.manager import Manager
from aegis.modules.initialization.terminalmanager import parse_terminal

from visor import visor


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
    custom_config_path, pickle_path, overwrite, run_visor = parse_terminal()

    if run_visor or (custom_config_path, pickle_path, overwrite) == (None, None, None):
        logging.info("Starting run_visor.")
        visor.run()
    else:
        manager = Manager(
            custom_config_path=custom_config_path,
            pickle_path=pickle_path,
            overwrite=overwrite,
            custom_input_params={},
        )
        manager.run()


def run_from_server_visor():
    pass


def run_from_local_visor():
    visor.run()
