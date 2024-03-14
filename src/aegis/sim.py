import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s -- %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
    level=logging.DEBUG,
)


def run_sim(config_path, pickle_path, overwrite):

    from aegis import pan

    pan.init(config_path, pickle_path, overwrite)

    from aegis.modules.genetics import gstruc, flipmap, phenomap, interpreter, mutator

    gstruc.init()
    flipmap.init(pan.cnf.FLIPMAP_CHANGE_RATE, gstruc.get_shape())
    phenomap.init(pan.cnf.PHENOMAP_SPECS, pan.cnf.PHENOMAP_METHOD)
    interpreter.init(pan.cnf.BITS_PER_LOCUS, pan.cnf.DOMINANCE_FACTOR, pan.cnf.THRESHOLD)
    mutator.init(MUTATION_METHOD=pan.cnf.MUTATION_METHOD, MUTATION_RATIO=pan.cnf.MUTATION_RATIO)

    # Create ecosystem
    # Cannot import before
    from aegis.ecosystem import Ecosystem
    from aegis.modules import recorder

    if not pan.pickle_path:
        ecosystem = Ecosystem()
    else:
        from aegis.modules.population import Population

        population = Population.load_pickle_from(pan.pickle_path)
        ecosystem = Ecosystem(population)

    # Record input summary
    recorder.record_input_summary()

    # Run simulation (if population not extinct)
    while pan.var.stage <= pan.cnf.STAGES_PER_SIMULATION and not recorder.extinct:
        recorder._log_progress(len(ecosystem.population))
        ecosystem.run_stage()
        pan.var.stage += 1

    # Record output summary
    recorder.record_output_summary()
    if recorder.extinct:
        logging.info("Population went extinct")
    else:
        logging.info("Simulation is successfully finished")
