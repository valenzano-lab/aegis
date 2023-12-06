"""This script is executed when you run `python3 -m aegis`."""

import logging
import pickle

from aegis import cnf
from visor import visor
from aegis import pan
from aegis import var


def main():
    if pan.config_path == "":
        visor.run()
    else:
        # Create ecosystem
        # Cannot import before
        from aegis.ecosystem import Ecosystem
        from aegis.modules import recorder

        if not pan.pickle_path:
            ecosystem = Ecosystem()
        else:
            with open(pan.pickle_path, "rb") as file_:
                population = pickle.load(file_)
                ecosystem = Ecosystem(population)

        # Record input summary
        recorder.record_input_summary()

        # Run simulation
        while var.stage <= cnf.STAGES_PER_SIMULATION_:
            recorder._log_progress()
            ecosystem.run_stage()
            var.stage += 1

        # Record output summary
        recorder.record_output_summary()
        logging.info("Simulation is successfully finished")


if __name__ == "__main__":
    main()
