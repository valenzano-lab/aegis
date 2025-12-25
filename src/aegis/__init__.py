"""
This script is executed when AEGIS is imported (`import aegis`). 
"""

import logging
import pathlib

import aegis_gui
import aegis_sim
from aegis.log import set_logging
from aegis.args_parsing import AegisArgumentParser


def start_from_terminal():
    parser = AegisArgumentParser()

    # Parse and validate arguments
    args = parser.parse_and_validate()

    set_logging(level=logging.INFO)
    logging.getLogger("numba").setLevel(logging.ERROR)

    if args.command == "sim":
        aegis_sim.run(
            custom_config_path=process_config_path(args.config_path),
            pickle_path=process_pickle_paths(args.pickle_path),
            overwrite=args.overwrite,
            custom_input_params={},
            pickle_weights=args.pickle_weights,
        )
    elif args.command == "gui":
        if args.server:
            aegis_gui.run(environment="server", debug=False)
        else:
            aegis_gui.run(environment="local", debug=args.debug)
    else:
        parser.print_help()

def process_config_path(config_path):
    return pathlib.Path(config_path).absolute() if config_path else None

def process_pickle_paths(raw_pickle_paths):
    return [pathlib.Path(path).absolute() for path in raw_pickle_paths]