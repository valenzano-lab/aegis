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


def parse_overrides(override_args):
    """Turn ["KEY=VALUE", ...] into a typed dict, coerced to each parameter's dtype.

    Values arrive from the shell as strings, so they are cast using the declared dtype
    in DEFAULT_PARAMETERS. Booleans accept true/false/1/0 in any case. "None" maps to
    None, which the validator accepts only for parameters that actually default to None
    (STARVATION_MORTALITY_FACTOR, CARRYING_CAPACITY_EGGS, ...) and rejects otherwise.
    For an uncapped resource pool pass RESOURCE_MAXIMUM_AMOUNT=inf, not None.
    """
    from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS

    overrides = {}
    for item in override_args or []:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in DEFAULT_PARAMETERS:
            raise ValueError(f"'{key}' is not a valid parameter name")

        if raw == "None":
            overrides[key] = None
            continue
        dtype = DEFAULT_PARAMETERS[key].dtype
        if dtype is bool:
            if raw.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key} expects a boolean, got {raw!r}")
            overrides[key] = raw.lower() in ("true", "1")
        else:
            try:
                overrides[key] = dtype(raw)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{key} expects {dtype.__name__}, got {raw!r}") from e
    return overrides


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

        if args.override and not args.resume:
            print(
                "Error: --override can only be used with --resume (-r). "
                "For a fresh run, set the parameter in the config file.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            resume_overrides = parse_overrides(args.override)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
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
                resume_overrides=resume_overrides,
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
