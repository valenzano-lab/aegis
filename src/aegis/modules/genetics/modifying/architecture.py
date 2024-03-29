import numpy as np

from aegis.modules.genetics.modifying.gpm_decoder import GPM_decoder
from aegis.modules.genetics.modifying.gpm import GPM


class ModifyingArchitecture:
    def __init__(self, ploid, PHENOMAP, MAX_LIFESPAN):
        self.ploid = ploid
        self.PHENOMAP = PHENOMAP
        self.MAX_LIFESPAN = MAX_LIFESPAN

        self.gpm_decoder = GPM_decoder(PHENOMAP)

        self.length = self.gpm_decoder.n

        self.phenomap = GPM(
            MAX_LIFESPAN=MAX_LIFESPAN,
            phenomatrix=None,
            phenolist=self.gpm_decoder.get_total_phenolist(),
        )

    def get_number_of_phenotypic_values(self):
        return self.MAX_LIFESPAN * 4

    def get_number_of_bits(self):
        return self.length

    def get_shape(self):
        return (self.ploid.y, self.length, 1)

    def init_genome_array(self, popsize):
        return np.zeros(shape=(popsize, *self.get_shape()), dtype=np.bool_)

    def init_phenotype_array(self, popsize):
        return np.zeros(shape=(popsize, self.get_number_of_phenotypic_values()))

    def compute(self, genomes):

        if genomes.shape[1] == 1:  # Do not calculate mean if genomes are haploid
            genomes = genomes[:, 0]
        else:
            genomes = self.ploid.diploid_to_haploid(genomes)

        return self.phenomap(
            interpretome=genomes.reshape(len(genomes), -1),
            zeropheno=self.init_phenotype_array(
                popsize=len(genomes),
            ),
        )
