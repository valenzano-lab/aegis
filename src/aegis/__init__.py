"""This script is executed when this package is imported."""

import logging
import pickle


def main(
    arg_dict=None,  # used when programmatically calling AEGIS
):
    from aegis.ecosystem import Ecosystem
    from aegis.panconfiguration import pan

    # Initialize pan
    pan.init(arg_dict)

    # Create ecosystems
    if not pan.pickle_path:
        ecosystems = [Ecosystem(i) for i in range(len(pan.params_list))]
    else:
        # TODO fix for multiple populations
        with open(pan.pickle_path, "rb") as file_:
            population = pickle.load(file_)
        ecosystem = Ecosystem(0, population)
        ecosystems = [ecosystem]

    # Run simulation
    while pan.stage <= pan.STAGES_PER_SIMULATION_:
        pan._log_progress()
        for ecosystem in ecosystems:
            ecosystem.run_stage()
        pan.stage += 1

    # Record output summary
    for ecosystem in ecosystems:
        ecosystem.recorder.record_output_summary()
    
    ecosystem.recorder.record_jupyter_path() # TODO record for every ecosystem

    logging.info("Simulation is successfully finished")
    logging.info("Custom jupyter path = %s", str(pan.output_path.absolute()))
    logging.info("Run visor by executing: python3 -m notebook %s", str(pan.here / "help"))
