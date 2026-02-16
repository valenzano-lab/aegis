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

    # Mutually exclusive: --resume vs (--config_path, --pickle_path, --overwrite)
    sim_mode = subparser_sim.add_mutually_exclusive_group()

    sim_mode.add_argument(
        "-r",
        "--resume",
        type=str,
        help="path to output directory to resume simulation from (uses latest checkpoint)",
        default=None,
    )

    sim_mode.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path to config file",
        default=None,
    )

    subparser_sim.add_argument(
        "-p",
        "--pickle_path",
        type=str,
        help="path to pickle file (seed mode)",
        default=None,
    )
    subparser_sim.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite old data with new simulation",
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
