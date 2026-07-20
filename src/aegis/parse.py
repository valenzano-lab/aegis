import argparse
import pathlib
import sys


def validate_config_path(path_str):
    """Validate that a config path has a proper stem (not just an extension)."""
    if path_str is None:
        return None
    p = pathlib.Path(path_str)
    if p.stem == "" or p.stem.startswith("."):
        print(f"Error: Config path '{path_str}' has no valid name (stem is empty or hidden).", file=sys.stderr)
        sys.exit(1)
    return path_str


def get_parser():

    parser = argparse.ArgumentParser(description="Aging of Evolving Genomes in Silico")
    subparsers = parser.add_subparsers(dest="command", help="")

    # subparser_sim
    subparser_sim = subparsers.add_parser("sim", help="run a simulation")

    subparser_sim.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path to config file (always required)",
        required=True,
    )

    # -o, -p, -r are mutually exclusive
    mode_group = subparser_sim.add_mutually_exclusive_group()

    mode_group.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite old data with new simulation",
        default=False,
    )

    mode_group.add_argument(
        "-p",
        "--pickle_path",
        type=str,
        help="path to pickle file (seed mode — new sim from saved population)",
        default=None,
    )

    mode_group.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="resume simulation from latest checkpoint",
        default=False,
    )

    subparser_sim.add_argument(
        "--extend",
        type=int,
        help="extend a resumed simulation to this many total steps",
        default=None,
    )

    subparser_sim.add_argument(
        "--override",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "override a parameter when resuming (repeatable), e.g. "
            "--override RESOURCE_MAXIMUM_AMOUNT=5000. Lets one burnt-in checkpoint be "
            "resumed under several regimes. Parameters defining genome/array shape "
            "(AGE_LIMIT, BITS_PER_LOCUS, PLOIDY, ...) are rejected."
        ),
        default=None,
    )

    # subparser_gui
    subparser_gui = subparsers.add_parser("gui", help="run GUI")
    subparser_gui.add_argument(
        "--server",
        "-s",
        action="store_true",
        help="run gui – the interactive GUI – in server mode",
        default=False,
    )

    subparser_gui.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="activate the debugger if running locally",
        default=False,
    )

    return parser
