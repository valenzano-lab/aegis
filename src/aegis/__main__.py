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

        # Run simulation (if population not extinct)
        while var.stage <= cnf.STAGES_PER_SIMULATION_ and not recorder.extinct:
            recorder._log_progress(len(ecosystem.population))
            ecosystem.run_stage()
            var.stage += 1

        # Record output summary
        recorder.record_output_summary()
        if recorder.extinct:
            logging.info("Population went extinct")
        else:
            logging.info("Simulation is successfully finished")


if __name__ == "__main__":
    main()
