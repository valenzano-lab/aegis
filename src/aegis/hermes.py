"""
Hermes acts as a central messenger, a repository of information that is to be used globally.
"""

import types
import logging
import numpy as np

from aegis import constants
from aegis.modules.initialization.parameterization.parametermanager import ParameterManager


# TODO resolve logging customization / logger
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s -- %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
    level=logging.DEBUG,
)


class Hermes:
    """ """

    def initialize(self, custom_config_path, custom_input_params, overwrite):
        self.stage = 1
        self.constants = constants

        self.parameters = ParameterManager(
            custom_config_path=custom_config_path,
            custom_input_params=custom_input_params,
        )()

        self.random_seed = self.init_random_seed()
        self.rng = self.init_random_number_generator()
        self.traits = self.init_traits()
        self.modules = self.init_modules()
        self.recorder = self.init_recorder(custom_config_path=custom_config_path, overwrite=overwrite)
        # self.logger = self.init_logger()  # TODO

    ##############
    # INIT FUNCS #
    ##############

    def init_random_seed(self):
        if self.parameters.RANDOM_SEED is None:
            return np.random.randint(1, 10**6)
        else:
            return self.parameters.RANDOM_SEED

    def init_random_number_generator(self):
        return np.random.default_rng(self.random_seed)

    def init_traits(self):
        """
        Here the trait order is hardcoded.
        """
        from aegis.modules.initialization.parameterization.trait import Trait

        traits = {}
        for traitname in constants.EVOLVABLE_TRAITS:
            trait = Trait(name=traitname, cnf=self.parameters)
            traits[traitname] = trait
        return traits

    def init_recorder(self, custom_config_path, overwrite):
        # NOTE Circular import if put on top
        from aegis.modules.recording.recordingmanager import RecordingManager

        return RecordingManager(custom_config_path, overwrite)

    def init_modules(self):
        # NOTE Circular import if put on top
        from aegis.modules.reproduction.mutation import Mutator
        from aegis.modules.reproduction.reproduction import Reproducer
        from aegis.modules.mortality.abiotic import Abiotic
        from aegis.modules.mortality.predation import Predation
        from aegis.modules.mortality.starvation import Starvation
        from aegis.modules.mortality.infection import Infection
        from aegis.modules.genetics.ploider import Ploider
        from aegis.modules.genetics.architect import Architect
        from aegis.utilities.popgenstats import PopgenStats

        modules = types.SimpleNamespace()
        # Mortality
        modules.abiotic = Abiotic(
            ABIOTIC_HAZARD_SHAPE=self.parameters.ABIOTIC_HAZARD_SHAPE,
            ABIOTIC_HAZARD_OFFSET=self.parameters.ABIOTIC_HAZARD_OFFSET,
            ABIOTIC_HAZARD_AMPLITUDE=self.parameters.ABIOTIC_HAZARD_AMPLITUDE,
            ABIOTIC_HAZARD_PERIOD=self.parameters.ABIOTIC_HAZARD_PERIOD,
        )
        modules.predation = Predation(
            PREDATOR_GROWTH=self.parameters.PREDATOR_GROWTH,
            PREDATION_RATE=self.parameters.PREDATION_RATE,
        )
        modules.starvation = Starvation(
            STARVATION_RESPONSE=self.parameters.STARVATION_RESPONSE,
            STARVATION_MAGNITUDE=self.parameters.STARVATION_MAGNITUDE,
            CLIFF_SURVIVORSHIP=self.parameters.CLIFF_SURVIVORSHIP,
            CARRYING_CAPACITY=self.parameters.CARRYING_CAPACITY,
        )
        modules.infection = Infection(
            BACKGROUND_INFECTIVITY=self.parameters.BACKGROUND_INFECTIVITY,
            TRANSMISSIBILITY=self.parameters.TRANSMISSIBILITY,
            RECOVERY_RATE=self.parameters.RECOVERY_RATE,
            FATALITY_RATE=self.parameters.FATALITY_RATE,
        )

        # Reproduction
        modules.mutator = Mutator(
            MUTATION_RATIO=self.parameters.MUTATION_RATIO,
            MUTATION_METHOD=self.parameters.MUTATION_METHOD,
            MUTATION_AGE_MULTIPLIER=self.parameters.MUTATION_AGE_MULTIPLIER,
        )
        modules.reproduction = Reproducer(
            RECOMBINATION_RATE=self.parameters.RECOMBINATION_RATE,
            REPRODUCTION_MODE=self.parameters.REPRODUCTION_MODE,
            mutator=modules.mutator,
        )

        # Genetic architecture
        modules.ploidy = Ploider(
            REPRODUCTION_MODE=self.parameters.REPRODUCTION_MODE,
            DOMINANCE_FACTOR=self.parameters.DOMINANCE_FACTOR,
        )
        modules.architect = Architect(
            ploid=modules.ploidy,
            BITS_PER_LOCUS=self.parameters.BITS_PER_LOCUS,
            PHENOMAP=self.parameters.PHENOMAP,
            AGE_LIMIT=self.parameters.AGE_LIMIT,
            THRESHOLD=self.parameters.THRESHOLD,
            ENVDRIFT_RATE=self.parameters.ENVDRIFT_RATE,
        )

        # Other
        modules.popgenstats = PopgenStats()

        return modules

    #############
    # UTILITIES #
    #############

    def get_stage(self) -> int:
        return self.stage

    def increment_stage(self) -> None:
        self.stage += 1

    def skip(self, rate_name) -> bool:
        """Should you skip an action performed at a certain rate"""

        rate = getattr(self.parameters, rate_name)

        # Skip if rate deactivated
        if rate <= 0:
            return True

        # Do not skip first stage
        if self.stage == 1:
            return False

        # Skip unless stage is divisible by rate
        return self.stage % rate > 0


hermes = Hermes()
