import numpy as np
from aegis import pan
from aegis.modules.genetics.mutation import Mutator
from aegis.modules.genetics.gstruc import Gstruc
from aegis.modules.genetics.flipmap import Flipmap
from aegis.modules.genetics.phenomap import Phenomap
from aegis.modules.genetics.interpreter import Interpreter
from aegis.modules.mortality.abiotic import Abiotic
from aegis.modules.mortality.predation import Predation
from aegis.modules.mortality.starvation import Starvation
from aegis.modules.mortality.infection import Infection
from aegis.modules.genetics.reproduction import Reproduction

"""
Initialization dependencies:
- flipmap requires gstruc
"""

mutator = Mutator(
    MUTATION_RATIO=pan.cnf.MUTATION_RATIO,
    MUTATION_METHOD=pan.cnf.MUTATION_METHOD,
)
gstruc = Gstruc(
    REPRODUCTION_MODE=pan.cnf.REPRODUCTION_MODE,
    BITS_PER_LOCUS=pan.cnf.BITS_PER_LOCUS,
)
interpreter = Interpreter(
    pan.cnf.BITS_PER_LOCUS,
    pan.cnf.DOMINANCE_FACTOR,
    pan.cnf.THRESHOLD,
)
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

# modules with dependencies
flipmap = Flipmap(
    pan.cnf.FLIPMAP_CHANGE_RATE,
    gstruc.get_shape(),
)
phenomap = Phenomap(
    pan.cnf.PHENOMAP_SPECS,
    pan.cnf.PHENOMAP_METHOD,
    gstruc=gstruc,
)
reproduction = Reproduction(
    RECOMBINATION_RATE=pan.cnf.RECOMBINATION_RATE,
    REPRODUCTION_MODE=pan.cnf.REPRODUCTION_MODE,
    mutator=mutator,
)


def get_phenotypes(genomes):
    """Translate genomes into an array of phenotypes probabilities."""
    # Apply the flipmap
    envgenomes = flipmap.call(genomes.get_array())

    # Apply the interpreter functions
    interpretome = np.zeros(shape=(envgenomes.shape[0], envgenomes.shape[2]), dtype=np.float32)
    for trait in gstruc.get_evolvable_traits():
        loci = envgenomes[:, :, trait.slice]  # fetch
        probs = interpreter.call(loci, trait.interpreter)  # interpret
        interpretome[:, trait.slice] += probs  # add back

    # Apply phenomap
    phenotypes = phenomap.call(interpretome)

    # Apply lo and hi bound
    for trait in gstruc.get_evolvable_traits():
        lo, hi = trait.lo, trait.hi
        phenotypes[:, trait.slice] = phenotypes[:, trait.slice] * (hi - lo) + lo

    return phenotypes


def get_evaluation(population, attr, part=None):
    """
    Get phenotypic values of a certain trait for certain individuals.
    Note that the function returns 0 for individuals that are to not be evaluated.
    """
    which_individuals = np.arange(len(population))
    if part is not None:
        which_individuals = which_individuals[part]

    # first scenario
    trait = gstruc.get_trait(attr)
    if not trait.evolvable:
        probs = trait.initial

    # second and third scenario
    if trait.evolvable:
        which_loci = trait.start
        if trait.agespecific:
            which_loci += population.ages[which_individuals]

        probs = population.phenotypes[which_individuals, which_loci]

    # expand values back into an array with shape of whole population
    final_probs = np.zeros(len(population), dtype=np.float32)
    final_probs[which_individuals] += probs

    return final_probs
