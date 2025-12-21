import logging
import pathlib

from aegis_sim.dataclasses.population import Population
from aegis_sim.bioreactor import Bioreactor
from aegis_sim import variables, submodels, parameterization
from aegis_sim.parameterization import parametermanager
from aegis_sim.recording import recordingmanager


def run(custom_config_path, pickle_path, overwrite, custom_input_params, pickle_weights=None):
    init(custom_config_path, overwrite, pickle_path, custom_input_params)

    if pickle_path:
        # Ensure backward compatibility: convert single path to list
        if not isinstance(pickle_path, list):
            pickle_path = [pickle_path]
        logging.info(f"Initializing a population using pre-evolved populations from: {pickle_path}")
        population = load_pre_evolved(pickle_path, pickle_weights)
    else:
        logging.info("Initializing new population. Not using pre-evolved populations")
        population = Population.initialize(
            n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
            AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
        )
        if parametermanager.parameters.ORIGIN_TRACKING == "population_level":
            population.reset_origins(origin_tracking_number=1)

    bioreactor = Bioreactor(population)

    sim(bioreactor=bioreactor)


def init(custom_config_path, overwrite=False, pickle_path=None, custom_input_params={}):
    """

    When testing aegis, initialize all modules using this function, e.g.

    import aegis_sim
    aegis_sim.init("_.yml")

    And then you can safely import any module.
    """

    custom_config_path = pathlib.Path(custom_config_path)

    parametermanager.init(
        custom_config_path=custom_config_path,
        custom_input_params=custom_input_params,
    )
    variables.init(
        variables,
        custom_config_path=custom_config_path,
        pickle_path=pickle_path,
        RANDOM_SEED=parametermanager.parameters.RANDOM_SEED,
    )
    parameterization.init_traits(parameterization)
    submodels.init(submodels, parametermanager=parametermanager)

    recordingmanager.init(custom_config_path, overwrite)
    recordingmanager.initialize_recorders(TICKER_RATE=parametermanager.parameters.TICKER_RATE)


def sim(bioreactor):
    # presim
    recordingmanager.configrecorder.write_final_config_file(parametermanager.final_config)
    recordingmanager.ticker.start_process()
    ticker_pid = recordingmanager.ticker.pid
    assert ticker_pid is not None
    recordingmanager.summaryrecorder.write_input_summary(ticker_pid=recordingmanager.ticker.pid)
    # TODO hacky solution of decrementing and incrementing steps
    variables.steps -= 1
    recordingmanager.featherrecorder.write(bioreactor.population)
    variables.steps += 1

    # sim
    recordingmanager.phenomaprecorder.write()

    while (variables.steps <= parametermanager.parameters.STEPS_PER_SIMULATION) and not recordingmanager.is_extinct():
        recordingmanager.progressrecorder.write(len(bioreactor.population))
        recordingmanager.simpleprogressrecorder.write()
        bioreactor.run_step()
        variables.steps += 1

    # postsim
    recordingmanager.summaryrecorder.write_output_summary()
    logging.info("Simulation finished.")
    recordingmanager.ticker.stop_process()

def load_pre_evolved(pickle_paths, pickle_weights=None) -> Population:
    """
    Load and combine pre-evolved populations from pickle files.
    
    :param pickle_paths: List of pathlib.Path objects pointing to pickle files.
    :param pickle_weights: List of float weights for combining populations. Defaults to 1.
    """
    if pickle_weights is None:
        pickle_weights = [1.0] * len(pickle_paths)
    
    # Load and sample populations
    populations = [
        Population.load_pickle_from(path).sample(weight)
        for path, weight in zip(pickle_paths, pickle_weights)
    ]
    logging.info(f"Loaded {len(populations)} pre-evolved populations from pickle files.")

    # Ensure all populations have origins reset
    if parametermanager.parameters.ORIGIN_TRACKING == "population_level":
        for origin_tracking_number, population in enumerate(populations):
            population.reset_origins(origin_tracking_number=origin_tracking_number)

    logging.info(f"Reset origins for all loaded populations with shape: {population.origins.shape()}")
    # Combine populations into one
    combined_population = populations[0]
    for population in populations[1:]:
        combined_population += population
    return combined_population
