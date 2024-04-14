from aegis.modules.dataclasses.genomes import Genomes
from aegis.modules.reproduction.assortment import assortment
from aegis.modules.reproduction.recombination import recombination


class Reproducer:
    """
    VISOR
    Individuals are fertile starting with MATURATION_AGE (can be 0) until MENOPAUSE (if 0, no menopause occurs).
    Reproduction can be sexual (with diploid genomes) or asexual (with diploid or haploid genomes).
    When reproduction is sexual, recombination occurs in gametes at a rate of RECOMBINATION_RATE
    and gametes will inherit mutations at an age-independent rate
    which can be parameterized (genetics-independent) or set to evolve (genetics-dependent).
    Mutations cause the offspring genome bit states to flip from 0-to-1 or 1-to-0.
    The ratio of 0-to-1 and 1-to-0 can be modified using the MUTATION_RATIO.
    """

    # TODO probably better to split mutation logic into another domain and cluster together with genetic architecture stuff

    def __init__(self, RECOMBINATION_RATE, REPRODUCTION_MODE, mutator):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.mutator = mutator

    def generate_offspring_genomes(self, genomes, muta_prob, ages):

        if self.REPRODUCTION_MODE == "sexual":
            genomes = recombination(genomes, self.RECOMBINATION_RATE)
            genomes, _ = assortment(Genomes(genomes))

        genomes = self.mutator._mutate(genomes, muta_prob, ages)
        genomes = Genomes(genomes)
        return genomes
