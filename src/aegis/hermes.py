"""
Hermes acts as a central messenger, a repository of information that is to be used globally.
"""

from typing import Optional
import logging
import numpy as np

from aegis import constants
from aegis.modules.initialization.parameterization.parametermanager import ParameterManager


# TODO resolve logging customization / logger
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s: %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
    level=logging.DEBUG,
)


class Hermes:
    """ """

    def initialize(self, custom_config_path, custom_input_params, overwrite):
        self.step = 1
        self.constants = constants

        parametermanager = ParameterManager(
            custom_config_path=custom_config_path,
            custom_input_params=custom_input_params,
        )
        self.parameters = parametermanager()

        self.simname = custom_config_path.stem
        self.random_seed = self.init_random_seed()
        self.rng = self.init_random_number_generator()
        self.traits = self.init_traits()
        self.modules = self.init_modules()
        self.recording_manager = self.init_recorder(custom_config_path=custom_config_path, overwrite=overwrite)
        self.recording_manager.configrecorder.write_final_config_file(parametermanager.final_config)

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

        recorder_manager = RecordingManager(custom_config_path, overwrite)
        recorder_manager.initialize_recorders(TICKER_RATE=self.parameters.TICKER_RATE)
        return recorder_manager

    def init_modules(self):
        # NOTE Circular import if put on top
        # NOTE when modules are put into simplenamespace, no code intelligence can be used
        from aegis.modules.reproduction.mutation import Mutator
        from aegis.modules.reproduction.reproduction import Reproducer
        from aegis.modules.mortality.abiotic import Abiotic
        from aegis.modules.mortality.predation import Predation
        from aegis.modules.mortality.starvation import Starvation
        from aegis.modules.mortality.infection import Infection
        from aegis.modules.genetics.ploider import Ploider
        from aegis.modules.genetics.architect import Architect
        from aegis.utilities.popgenstats import PopgenStats
        from aegis.modules.resources.resources import Resources
        from aegis.modules.reproduction.sexsystem import SexSystem
        from aegis.modules.reproduction.matingmanager import MatingManager

        class Modules:
            def __init__(
                self,
                abiotic: Optional[Abiotic] = None,
                predation: Optional[Predation] = None,
                starvation: Optional[Starvation] = None,
                infection: Optional[Infection] = None,
                resources: Optional[Resources] = None,
                mutator: Optional[Mutator] = None,
                reproduction: Optional[Reproducer] = None,
                matingmanager: Optional[MatingManager] = None,
                ploidy: Optional[Ploider] = None,
                sexsystem: Optional[SexSystem] = None,
                popgenstats: Optional[PopgenStats] = None,
            ):
                self.abiotic = abiotic
                self.predation = predation
                self.starvation = starvation
                self.infection = infection
                self.resources = resources
                self.mutator = mutator
                self.reproduction = reproduction
                self.matingmanager = matingmanager
                self.ploidy = ploidy
                self.sexsystem = sexsystem
                self.popgenstats = popgenstats

        # modules = types.SimpleNamespace(
        #     abiotic=None  # type: typing.Optional[Abiotic],
        # )
        modules = Modules()
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

        # Resources
        modules.resources = Resources(
            CARRYING_CAPACITY=self.parameters.CARRYING_CAPACITY,
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

        modules.matingmanager = MatingManager()

        # Genetic architecture
        modules.ploidy = Ploider(
            REPRODUCTION_MODE=self.parameters.REPRODUCTION_MODE,
            DOMINANCE_FACTOR=self.parameters.DOMINANCE_FACTOR,
        )
        self.architect = Architect(
            ploid=modules.ploidy,
            BITS_PER_LOCUS=self.parameters.BITS_PER_LOCUS,
            PHENOMAP=self.parameters.PHENOMAP,
            AGE_LIMIT=self.parameters.AGE_LIMIT,
            THRESHOLD=self.parameters.THRESHOLD,
            ENVDRIFT_RATE=self.parameters.ENVDRIFT_RATE,
        )
        modules.sexsystem = SexSystem()

        # Other
        modules.popgenstats = PopgenStats()

        return modules

    #############
    # UTILITIES #
    #############

    def get_step(self) -> int:
        return self.step

    def increment_step(self) -> None:
        self.step += 1

    def skip(self, rate_name) -> bool:
        """Should you skip an action performed at a certain rate"""

        rate = getattr(self.parameters, rate_name)

        # Skip if rate deactivated
        if rate <= 0:
            return True

        # Do not skip first step
        if self.step == 1:
            return False

        # Skip unless step is divisible by rate
        return self.step % rate > 0

    def steps_to_end(self) -> int:
        return self.parameters.STEPS_PER_SIMULATION - self.step


hermes = Hermes()
