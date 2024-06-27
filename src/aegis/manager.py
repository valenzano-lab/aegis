import logging  # TODO use logger
from aegis.hermes import hermes
from aegis.modules.dataclasses.population import Population
from aegis.bioreactor import Bioreactor


class Manager:
    def __init__(self, custom_config_path, pickle_path, overwrite, custom_input_params):

        # Inputs
        self.custom_config_path = custom_config_path
        self.pickle_path = pickle_path
        self.overwrite = overwrite
        self.custom_input_params = custom_input_params

        # Main container of simulation objects and logic
        self.bioreactor = None

    def run(self) -> None:
        logging.info(f"Running {self.custom_config_path}.")
        self.run_initialization()
        self.run_simulation()

    ##########################
    # INITIALIZATION METHODS #
    ##########################

    def run_initialization(self) -> None:
        hermes.initialize(
            custom_config_path=self.custom_config_path,
            custom_input_params=self.custom_input_params,
            overwrite=self.overwrite,
        )
        population = self.initialize_population(self.pickle_path)
        self.bioreactor = self.initialize_bioreactor(population)

    @staticmethod
    def initialize_population(pickle_path):
        if pickle_path is None:  # Fresh population
            return Population.initialize(n=hermes.parameters.CARRYING_CAPACITY)
        else:  # Pickled population
            return Population.load_pickle_from(pickle_path)

    @staticmethod
    def initialize_bioreactor(population):
        return Bioreactor(population)

    ######################
    # SIMULATION METHODS #
    ######################

    def run_simulation(self) -> None:
        hermes.recording_manager.ticker.start_process()
        self.log_pre_simulation()
        self.run_simulation_steps()
        self.log_post_simulation()
        hermes.recording_manager.ticker.stop_process()

    @staticmethod
    def log_pre_simulation():
        hermes.recording_manager.summaryrecorder.write_input_summary()

    def run_simulation_steps(self) -> None:
        hermes.recording_manager.phenomaprecorder.write()
        while not self.is_finished() and not self.is_extinct():
            hermes.recording_manager.progressrecorder.write(len(self.bioreactor.population))
            self.bioreactor.run_step()
            hermes.increment_step()

    @staticmethod
    def log_post_simulation() -> None:
        hermes.recording_manager.summaryrecorder.write_output_summary()
        logging.info("Simulation finished.")

    def is_extinct(self) -> bool:
        if hermes.recording_manager.summaryrecorder.extinct:
            logging.info(f"Population went extinct (at step {hermes.step}).")
            return True
        return False

    def is_finished(self) -> bool:
        if hermes.get_step() <= hermes.parameters.STEPS_PER_SIMULATION:
            return False
        logging.info("All simulation steps successfully simulated.")
        return True
