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
        config_path_str = validate_config_path(args.config_path)
        config_path = pathlib.Path(config_path_str).absolute()

        if args.extend is not None and not args.resume:
            print("Error: --extend can only be used with --resume (-r).", file=sys.stderr)
            sys.exit(1)

        if args.resume:
            # Resume mode — derive output dir from config path
            odir = config_path.parent / config_path.stem
            aegis_sim.run(
                custom_config_path=config_path,
                pickle_path=None,
                overwrite=False,
                custom_input_params={},
                resume_path=odir,
                extend_steps=args.extend,
            )
        else:
            # Fresh or seed mode
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
