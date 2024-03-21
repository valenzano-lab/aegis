from aegis.modules.dataclasses.genomes import Genomes
from aegis.modules.reproduction.assortment import assortment
from aegis.modules.reproduction.recombination import recombination


class Reproducer:
    def __init__(self, RECOMBINATION_RATE, REPRODUCTION_MODE, mutator):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.mutator = mutator

    def generate_offspring_genomes(self, genomes, muta_prob):

        if self.REPRODUCTION_MODE == "sexual":
            genomes = recombination(genomes, self.RECOMBINATION_RATE)
            genomes, _ = assortment(Genomes(genomes))

        genomes = self.mutator._mutate(genomes, muta_prob)
        genomes = Genomes(genomes)
        return genomes
