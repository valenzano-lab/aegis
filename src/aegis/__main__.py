"""This script is executed when you run `python3 -m aegis`."""

import logging
import pickle
import argparse
import pathlib
from aegis.visor import visor


def parse():
    # Parse terminal arguments

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
        type=bool,
        help="overwrite old data with new simulation",
        default=False,
    )

    # Parse arguments
    args = parser.parse_args()

    # Decision tree
    config_path = pathlib.Path(args.config_path).absolute() if args.config_path else ""
    pickle_path = pathlib.Path(args.pickle_path).absolute() if args.pickle_path else ""

    return config_path, pickle_path, args.overwrite


def main():
    from aegis.ecosystem import Ecosystem
    from aegis import pan

    config_path, pickle_path, overwrite = parse()

    if config_path:
        # Initialize pan
        pan.init(config_path, overwrite, running_on_server=False)

        # Create ecosystem
        if not pickle_path:
            ecosystem = Ecosystem()
        else:
            with open(pickle_path, "rb") as file_:
                population = pickle.load(file_)
                ecosystem = Ecosystem(population)

        # Record input summary
        ecosystem.recorder.record_input_summary()

        # Run simulation
        while pan.stage <= pan.STAGES_PER_SIMULATION_:
            pan._log_progress()
            ecosystem.run_stage()
            pan.stage += 1

        # Record output summary
        ecosystem.recorder.record_output_summary()
        logging.info("Simulation is successfully finished")
    else:
        visor.run()


if __name__ == "__main__":
    main()
