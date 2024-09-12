import argparse


def get_parser():

    parser = argparse.ArgumentParser(description="Aging of Evolving Genomes in Silico")
    subparsers = parser.add_subparsers(dest="command", help="")

    # subparser_sim
    subparser_sim = subparsers.add_parser("sim", help="run a simulation")
    subparser_sim.add_argument(
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
        help="path to pickle file",
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
