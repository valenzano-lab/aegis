import logging
import pathlib

from aegis_sim.dataclasses.individual import Group
from aegis_sim.bioreactor import Bioreactor
from aegis_sim import variables, submodels, parameterization
from aegis_sim.parameterization import parametermanager
from aegis_sim.recording import recordingmanager


def run(custom_config_path, pickle_path, overwrite, custom_input_params):
    init(custom_config_path, overwrite, pickle_path, custom_input_params)

    group = (
        Group.initialize(
            n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
            AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
        )
        if pickle_path is None
        else Group.load_pickle_from(pickle_path)
    )

    bioreactor = Bioreactor(group)

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
    recordingmanager.featherrecorder.write(bioreactor.group)
    variables.steps += 1

    # sim
    recordingmanager.phenomaprecorder.write()

    while (variables.steps <= parametermanager.parameters.STEPS_PER_SIMULATION) and not recordingmanager.is_extinct():
        recordingmanager.progressrecorder.write(len(bioreactor.group))
        recordingmanager.simpleprogressrecorder.write()
        bioreactor.run_step()
        variables.steps += 1

    # postsim
    recordingmanager.summaryrecorder.write_output_summary()
    logging.info("Simulation finished.")
    recordingmanager.ticker.stop_process()
