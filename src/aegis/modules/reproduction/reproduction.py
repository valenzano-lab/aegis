from aegis.modules.dataclasses.genomes import Genomes
from aegis.modules.reproduction.assortment import assortment
from aegis.modules.reproduction.recombination import recombination


class Reproducer:
    """
    VISOR
    Individuals are fertile starting with [[MATURATION_AGE]] (can be 0) until [[MENOPAUSE]] (if 0, no menopause occurs).
    Reproduction can be sexual (with diploid genomes) or asexual (with diploid or haploid genomes).
    When reproduction is sexual, recombination occurs in gametes at a rate of [[RECOMBINATION_RATE]]
    and gametes will inherit mutations at an age-independent rate
    which can be parameterized (genetics-independent) or set to evolve (genetics-dependent).
    Mutations cause the offspring genome bit states to flip from 0-to-1 or 1-to-0.
    The ratio of 0-to-1 and 1-to-0 can be modified using the [[MUTATION_RATIO]].

    If the population is oviparous, [[INCUBATION_PERIOD]] should be set to -1, 1 or greater.
    When it is set to -1, all laid eggs hatch only once all living individuals die.
    When it is set to 0 or greater, eggs hatch after that specified time.
    Thus, when 0, the population has no egg life step.
    """

    # TODO INCUBATION_PERIOD set to -1 or 1 or greater is stupid
    # TODO INCUBATION_PERIOD is not really a part of this submodel, but it is in the documentation. it should be though.

    # TODO probably better to split mutation logic into another domain and cluster together with genetic architecture stuff

    def __init__(self, RECOMBINATION_RATE, REPRODUCTION_MODE, mutator):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.mutator = mutator

    def generate_offspring_genomes(self, genomes, muta_prob, ages, parental_sexes):

        if self.REPRODUCTION_MODE == "sexual":
            genomes = recombination(genomes, self.RECOMBINATION_RATE)
            genomes = assortment(Genomes(genomes), parental_sexes)

        genomes = self.mutator._mutate(genomes, muta_prob, ages)
        genomes = Genomes(genomes)
        return genomes
