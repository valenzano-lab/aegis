"""This script is executed when you run `python3 -m aegis`."""


def parse():
    import argparse
    import pathlib

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


if __name__ == "__main__":

    config_path, pickle_path, overwrite = parse()
    if config_path == "":
        from visor import visor

        visor.run()
    else:
        from aegis.sim import run_sim

        run_sim(config_path, pickle_path, overwrite)
