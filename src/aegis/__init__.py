"""This script is executed when this package is imported."""

import logging


def main(custom_config_path=None):
    from aegis.ecosystem import Ecosystem
    from aegis.panconfiguration import pan

    # Initialize pan
    pan.init(custom_config_path)

    # Create ecosystems
    ecosystems = [Ecosystem(i) for i in range(len(pan.params_list))]

    # Run simulation
    while pan.run_stage():
        for ecosystem in ecosystems:
            ecosystem.run_stage()

    # Record output summary
    for ecosystem in ecosystems:
        ecosystem.recorder.record_output_summary()

    logging.info("Simulation is successfully finished")
