import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s -- %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
    level=logging.DEBUG,
)


def run_sim(config_path, pickle_path, overwrite):

    from aegis import pan

    pan.init(config_path, pickle_path, overwrite)

    # Create ecosystem
    # Cannot import before
    from aegis.ecosystem import Ecosystem
    from aegis.modules.recording import recorder

    if not pan.pickle_path:
        ecosystem = Ecosystem()
    else:
        from aegis.modules.dataclasses.population import Population

        population = Population.load_pickle_from(pan.pickle_path)
        ecosystem = Ecosystem(population)

    # Record input summary
    recorder.record_input_summary()

    # Run simulation (if population not extinct)
    while pan.get_stage() <= pan.cnf.STAGES_PER_SIMULATION and not recorder.extinct:
        recorder._log_progress(len(ecosystem.population))
        ecosystem.run_stage()
        pan.increment_stage()

    # Record output summary
    recorder.record_output_summary()
    if recorder.extinct:
        logging.info("Population went extinct")
    else:
        logging.info("Simulation is successfully finished")
