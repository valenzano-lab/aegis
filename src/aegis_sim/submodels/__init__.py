from aegis_sim.submodels.reproduction.mutation import Mutator
from aegis_sim.submodels.reproduction.reproduction import Reproducer
from aegis_sim.submodels.abiotic import Abiotic
from aegis_sim.submodels.predation import Predation
from aegis_sim.submodels.resources.starvation import Starvation
from aegis_sim.submodels.infection import Infection
from aegis_sim.submodels.frailty import Frailty
from aegis_sim.submodels.genetics.ploider import Ploider
from aegis_sim.submodels.genetics.architect import Architect
from aegis_sim.utilities.popgenstats import PopgenStats
from aegis_sim.submodels.resources.resources import Resources
from aegis_sim.submodels.reproduction.sexsystem import SexSystem
from aegis_sim.submodels.reproduction.matingmanager import MatingManager


def init(self, parametermanager):

    ##################################
    # INDEPENDENT of other submodels #
    ##################################

    self.abiotic = Abiotic(
        ABIOTIC_HAZARD_SHAPE=parametermanager.parameters.ABIOTIC_HAZARD_SHAPE,
        ABIOTIC_HAZARD_OFFSET=parametermanager.parameters.ABIOTIC_HAZARD_OFFSET,
        ABIOTIC_HAZARD_AMPLITUDE=parametermanager.parameters.ABIOTIC_HAZARD_AMPLITUDE,
        ABIOTIC_HAZARD_PERIOD=parametermanager.parameters.ABIOTIC_HAZARD_PERIOD,
    )
    self.predation = Predation(
        PREDATOR_GROWTH=parametermanager.parameters.PREDATOR_GROWTH,
        PREDATION_RATE=parametermanager.parameters.PREDATION_RATE,
    )
    self.starvation = Starvation(
        STARVATION_RESPONSE=parametermanager.parameters.STARVATION_RESPONSE,
        STARVATION_MAGNITUDE=parametermanager.parameters.STARVATION_MAGNITUDE,
        CLIFF_SURVIVORSHIP=parametermanager.parameters.CLIFF_SURVIVORSHIP,
        CARRYING_CAPACITY=parametermanager.parameters.CARRYING_CAPACITY,
    )
    self.infection = Infection(
        BACKGROUND_INFECTIVITY=parametermanager.parameters.BACKGROUND_INFECTIVITY,
        TRANSMISSIBILITY=parametermanager.parameters.TRANSMISSIBILITY,
        RECOVERY_RATE=parametermanager.parameters.RECOVERY_RATE,
        FATALITY_RATE=parametermanager.parameters.FATALITY_RATE,
    )
    self.frailty = Frailty(
        FRAILTY_MODIFIER=parametermanager.parameters.FRAILTY_MODIFIER,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )

    # Resources
    self.resources = Resources(
        CARRYING_CAPACITY=parametermanager.parameters.CARRYING_CAPACITY,
        RESOURCE_ADDITIVE_GROWTH=parametermanager.parameters.RESOURCE_ADDITIVE_GROWTH,
        RESOURCE_MULTIPLICATIVE_GROWTH=parametermanager.parameters.RESOURCE_MULTIPLICATIVE_GROWTH,
    )

    # Reproduction
    self.mutator = Mutator(
        MUTATION_RATIO=parametermanager.parameters.MUTATION_RATIO,
        MUTATION_METHOD=parametermanager.parameters.MUTATION_METHOD,
        MUTATION_AGE_MULTIPLIER=parametermanager.parameters.MUTATION_AGE_MULTIPLIER,
    )
    self.sexsystem = SexSystem()
    self.matingmanager = MatingManager()

    # Genetic architecture
    self.ploidy = Ploider(
        REPRODUCTION_MODE=parametermanager.parameters.REPRODUCTION_MODE,
        DOMINANCE_FACTOR=parametermanager.parameters.DOMINANCE_FACTOR,
        PLOIDY=parametermanager.parameters.PLOIDY,
    )

    # Other
    self.popgenstats = PopgenStats()

    ################################
    # DEPENDENT on other submodels #
    ################################

    # Reproduction
    self.reproduction = Reproducer(
        RECOMBINATION_RATE=parametermanager.parameters.RECOMBINATION_RATE,
        REPRODUCTION_MODE=parametermanager.parameters.REPRODUCTION_MODE,
        mutator=self.mutator,
    )

    # Genetic architecture
    self.architect = Architect(
        ploid=self.ploidy,
        BITS_PER_LOCUS=parametermanager.parameters.BITS_PER_LOCUS,
        PHENOMAP=parametermanager.parameters.PHENOMAP,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
        THRESHOLD=parametermanager.parameters.THRESHOLD,
        ENVDRIFT_RATE=parametermanager.parameters.ENVDRIFT_RATE,
    )
