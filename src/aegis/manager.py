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
            return Population.initialize(N=hermes.parameters.MAX_POPULATION_SIZE)
        else:  # Pickled population
            return Population.load_pickle_from(pickle_path)

    @staticmethod
    def initialize_bioreactor(population):
        return Bioreactor(population)

    ######################
    # SIMULATION METHODS #
    ######################

    def run_simulation(self) -> None:
        self.log_pre_simulation()
        self.run_simulation_stages()
        self.log_post_simulation()

    @staticmethod
    def log_pre_simulation():
        hermes.recorder.summaryrecorder.record_input_summary()

    def run_simulation_stages(self) -> None:
        hermes.recorder.phenomaprecorder.write()
        while not self.is_finished() and not self.is_extinct():
            hermes.recorder.progressrecorder.write(len(self.bioreactor.population))
            self.bioreactor.run_stage()
            hermes.increment_stage()

    @staticmethod
    def log_post_simulation() -> None:
        hermes.recorder.summaryrecorder.record_output_summary()
        logging.info("Simulation finished")

    def is_extinct(self) -> bool:
        if hermes.recorder.summaryrecorder.extinct:
            logging.info(f"Population went extinct (at stage {hermes.stage})")
            return True
        return False

    def is_finished(self) -> bool:
        if hermes.get_stage() <= hermes.parameters.STAGES_PER_SIMULATION:
            return False
        logging.info("All simulation stages successfully simulated")
        return True
