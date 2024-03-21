import numpy as np
from aegis.hermes import hermes

from aegis.modules.genetics.composite.interpreter import Interpreter


class CompositeArchitecture:
    def __init__(self, ploid, BITS_PER_LOCUS, MAX_LIFESPAN, THRESHOLD):
        self.ploid = ploid
        self.BITS_PER_LOCUS = BITS_PER_LOCUS
        self.n_loci = 4 * MAX_LIFESPAN
        self.MAX_LIFESPAN = MAX_LIFESPAN

        self.evolvable = [trait for trait in hermes.traits.values() if trait.evolvable]

        self.interpreter = Interpreter(
            self.BITS_PER_LOCUS,
            THRESHOLD,
        )

    def get_number_of_phenotypic_values(self):
        return self.MAX_LIFESPAN * 4

    def get_number_of_bits(self):
        return self.ploid.y * self.n_loci * self.BITS_PER_LOCUS

    def get_shape(self):
        return (self.ploid.y, self.n_loci, self.BITS_PER_LOCUS)

    def init_genome_array(self, popsize):
        # TODO enable initgeno
        array = hermes.rng.random(size=(popsize, *self.get_shape()), dtype=np.float32)

        for trait in hermes.traits.values():
            array[:, :, trait.slice] = array[:, :, trait.slice] < trait.initgeno

        return array

    def init_phenotype_array(self, popsize):
        return np.zeros(shape=(popsize, self.get_number_of_phenotypic_values()))

    def compute(self, genomes):

        if genomes.shape[1] == 1:  # Do not calculate mean if genomes are haploid
            genomes = genomes[:, 0]
        else:
            genomes = self.ploid.diploid_to_haploid(genomes)

        interpretome = np.zeros(shape=(genomes.shape[0], genomes.shape[1]), dtype=np.float32)
        for trait in hermes.traits.values():
            loci = genomes[:, trait.slice]  # fetch
            probs = self.interpreter.call(loci, trait.interpreter)  # interpret
            interpretome[:, trait.slice] += probs  # add back

        return interpretome
