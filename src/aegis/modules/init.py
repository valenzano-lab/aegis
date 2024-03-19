from aegis import pan

from aegis.modules.reproduction.mutation import Mutator
from aegis.modules.reproduction.reproduction import Reproducer

from aegis.modules.mortality.abiotic import Abiotic
from aegis.modules.mortality.predation import Predation
from aegis.modules.mortality.starvation import Starvation
from aegis.modules.mortality.infection import Infection

from aegis.modules.genetics.ploider import Ploider
from aegis.modules.genetics.architect import Architect


# Mortality
abiotic = Abiotic(
    ABIOTIC_HAZARD_SHAPE=pan.cnf.ABIOTIC_HAZARD_SHAPE,
    ABIOTIC_HAZARD_OFFSET=pan.cnf.ABIOTIC_HAZARD_OFFSET,
    ABIOTIC_HAZARD_AMPLITUDE=pan.cnf.ABIOTIC_HAZARD_AMPLITUDE,
    ABIOTIC_HAZARD_PERIOD=pan.cnf.ABIOTIC_HAZARD_PERIOD,
)
predation = Predation(
    PREDATOR_GROWTH=pan.cnf.PREDATOR_GROWTH,
    PREDATION_RATE=pan.cnf.PREDATION_RATE,
)
starvation = Starvation(
    STARVATION_RESPONSE=pan.cnf.STARVATION_RESPONSE,
    STARVATION_MAGNITUDE=pan.cnf.STARVATION_MAGNITUDE,
    CLIFF_SURVIVORSHIP=pan.cnf.CLIFF_SURVIVORSHIP,
    MAX_POPULATION_SIZE=pan.cnf.MAX_POPULATION_SIZE,
)
infection = Infection(
    BACKGROUND_INFECTIVITY=pan.cnf.BACKGROUND_INFECTIVITY,
    TRANSMISSIBILITY=pan.cnf.TRANSMISSIBILITY,
    RECOVERY_RATE=pan.cnf.RECOVERY_RATE,
    FATALITY_RATE=pan.cnf.FATALITY_RATE,
)


# Reproduction
mutator = Mutator(
    MUTATION_RATIO=pan.cnf.MUTATION_RATIO,
    MUTATION_METHOD=pan.cnf.MUTATION_METHOD,
)
reproduction = Reproducer(
    RECOMBINATION_RATE=pan.cnf.RECOMBINATION_RATE,
    REPRODUCTION_MODE=pan.cnf.REPRODUCTION_MODE,
    mutator=mutator,
)


# Genetic architecture
ploidy = Ploider(
    REPRODUCTION_MODE=pan.cnf.REPRODUCTION_MODE,
    DOMINANCE_FACTOR=pan.cnf.DOMINANCE_FACTOR,
)

architect = Architect(
    ploid=ploidy,
    BITS_PER_LOCUS=pan.cnf.BITS_PER_LOCUS,
    PHENOMAP=pan.cnf.PHENOMAP,
    MAX_LIFESPAN=pan.cnf.MAX_LIFESPAN,
    THRESHOLD=pan.cnf.THRESHOLD,
    FLIPMAP_CHANGE_RATE=pan.cnf.FLIPMAP_CHANGE_RATE,
)
