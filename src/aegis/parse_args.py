import argparse
import pathlib


def parse():
    # Define parser
    parser = argparse.ArgumentParser(description="Aging of Evolving Genomes in Silico")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path to config file",
        default="",
    )
    parser.add_argument(
        "-p",
        "--pickle_path",
        type=str,
        help="path to pickle file",
        default="",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="overwrite old data with new simulation",
        default=False,
    )

    # Parse arguments
    args = parser.parse_args()

    # Decision tree
    config_path = pathlib.Path(args.config_path).absolute() if args.config_path else ""
    pickle_path = pathlib.Path(args.pickle_path).absolute() if args.pickle_path else ""

    return config_path, pickle_path, args.overwrite
