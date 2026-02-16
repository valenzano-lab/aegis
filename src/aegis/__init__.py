"""
This script is executed when AEGIS is imported (`import aegis`). Execute functions by running `aegis.run_from_{}`.
AEGIS can be started in multiple ways; each of these functions starts AEGIS from a different context.
"""

import logging
import pathlib
import sys

import aegis_gui
import aegis_sim
from aegis.log import set_logging
from aegis.parse import get_parser, validate_config_path


def start_from_terminal():
    parser = get_parser()
    args = parser.parse_args()
    set_logging(level=logging.DEBUG)
    logging.getLogger("numba").setLevel(logging.ERROR)

    if args.command == "sim":
        resume_path = pathlib.Path(args.resume).absolute() if args.resume else None

        if resume_path is not None:
            # Resume mode — no config_path, pickle_path, or overwrite needed
            if args.pickle_path is not None:
                print("Error: --pickle_path cannot be used with --resume.", file=sys.stderr)
                sys.exit(1)
            if args.overwrite:
                print("Error: --overwrite cannot be used with --resume.", file=sys.stderr)
                sys.exit(1)
            aegis_sim.run(
                custom_config_path=None,
                pickle_path=None,
                overwrite=False,
                custom_input_params={},
                resume_path=resume_path,
            )
        else:
            # Fresh or seed mode
            config_path_str = validate_config_path(args.config_path)
            config_path = pathlib.Path(config_path_str).absolute() if config_path_str else None
            pickle_path = pathlib.Path(args.pickle_path).absolute() if args.pickle_path else None
            aegis_sim.run(
                custom_config_path=config_path,
                pickle_path=pickle_path,
                overwrite=args.overwrite,
                custom_input_params={},
            )
    elif args.command == "gui":
        if args.server:
            aegis_gui.run(environment="server", debug=False)
        else:
            aegis_gui.run(environment="local", debug=args.debug)
    else:
        parser.print_help()
