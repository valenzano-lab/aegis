from aegis_sim.submodels.reproduction.mutation import mutator
from aegis_sim.submodels.reproduction.reproduction import Reproducer
from aegis_sim.submodels.abiotic import Abiotic
from aegis_sim.submodels.predation import Predation
from aegis_sim.submodels.resources.starvation import starvation
from aegis_sim.submodels.infection import Infection
from aegis_sim.submodels.frailty import frailty
from aegis_sim.submodels.genetics.ploider import ploider
from aegis_sim.submodels.genetics.architect import Architect
from aegis_sim.utilities.popgenstats import PopgenStats
from aegis_sim.submodels.resources.resources import resources
from aegis_sim.submodels.reproduction.sexsystem import SexSystem
from aegis_sim.submodels.reproduction.matingmanager import MatingManager

# Type specification for Intellisense
architect: Architect
reproduction: Reproducer
sexsystem: SexSystem
matingmanager: MatingManager
infection: Infection
popgenstats: PopgenStats

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
    starvation.init(
        STARVATION_MORTALITY_FACTOR=parametermanager.parameters.STARVATION_MORTALITY_FACTOR,
        STARVATION_MORTALITY_MAXIMUM=parametermanager.parameters.STARVATION_MORTALITY_MAXIMUM,
    )
    self.infection = Infection(
        BACKGROUND_INFECTIVITY=parametermanager.parameters.BACKGROUND_INFECTIVITY,
        TRANSMISSIBILITY=parametermanager.parameters.TRANSMISSIBILITY,
        RECOVERY_RATE=parametermanager.parameters.RECOVERY_RATE,
        FATALITY_RATE=parametermanager.parameters.FATALITY_RATE,
    )
    frailty.init(
        FRAILTY_MODIFIER=parametermanager.parameters.FRAILTY_MODIFIER,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )

    # Resources
    resources.init(
        RESOURCE_ADDITIVE_GROWTH=parametermanager.parameters.RESOURCE_ADDITIVE_GROWTH,
        RESOURCE_MULTIPLICATIVE_GROWTH=parametermanager.parameters.RESOURCE_MULTIPLICATIVE_GROWTH,
        RESOURCE_MAXIMUM_AMOUNT=parametermanager.parameters.RESOURCE_MAXIMUM_AMOUNT,
        RESOURCE_INITIAL_AMOUNT=parametermanager.parameters.RESOURCE_INITIAL_AMOUNT,
    )

    # Reproduction
    mutator.init(
        MUTATION_RATIO=parametermanager.parameters.MUTATION_RATIO,
        MUTATION_METHOD=parametermanager.parameters.MUTATION_METHOD,
        MUTATION_AGE_MULTIPLIER=parametermanager.parameters.MUTATION_AGE_MULTIPLIER,
    )
    self.sexsystem = SexSystem()
    self.matingmanager = MatingManager()

    # Genetic architecture
    ploider.init(
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
        mutator=mutator,
    )

    # Genetic architecture
    self.architect = Architect(
        GENARCH_TYPE=parametermanager.parameters.GENARCH_TYPE,
        BITS_PER_LOCUS=parametermanager.parameters.BITS_PER_LOCUS,
        PHENOMAP=parametermanager.parameters.PHENOMAP,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
        THRESHOLD=parametermanager.parameters.THRESHOLD,
        ENVDRIFT_RATE=parametermanager.parameters.ENVDRIFT_RATE,
        MODIF_GENOME_SIZE=parametermanager.parameters.MODIF_GENOME_SIZE,
    )
