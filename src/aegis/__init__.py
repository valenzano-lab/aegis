__version__ = "2.3.0.18"  # TODO check if this is ok for setup.py

"""
This script is executed when AEGIS is imported (`import aegis`). Execute functions by running `aegis.run_from_{}`.
AEGIS can be started in multiple ways; each of these functions starts AEGIS from a different context. 
"""

import logging
import pathlib

import aegis_gui
import aegis_sim
from aegis.log import set_logging
from aegis.parse import get_parser


def start_from_terminal():
    parser = get_parser()
    args = parser.parse_args()
    set_logging(level=logging.DEBUG)

    if args.command == "sim":
        config_path = pathlib.Path(args.config_path).absolute() if args.config_path else None
        pickle_path = pathlib.Path(args.pickle_path).absolute() if args.pickle_path else None
        aegis_sim.run(
            custom_config_path=config_path,
            pickle_path=pickle_path,
            overwrite=args.overwrite,
            custom_input_params={},
        )
    elif args.command == "gui":
        if args.server:
            logging.info("Server mode is ON")
            aegis_gui.run(environment="server")
        else:
            logging.info("Server mode is OFF")
            aegis_gui.run(environment="local")
    else:
        parser.print_help()
