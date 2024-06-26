import numpy as np

from aegis import constants

from aegis.modules.genetics.modifying.gpm_decoder import GPM_decoder
from aegis.modules.genetics.modifying.gpm import GPM


class ModifyingArchitecture:
    """

    VISOR
    - when pleiotropy is needed
    - when all bits are 0, the phenotypic values are the ones set from parameters (baseline set in parameters);
    vs composite where it would be 0.
    - ... dev still required

    
    ### MODIFYING GENETIC ARCHITECTURE (MGA) ###

    
    A modifying genetic architecture (MGA) is useful when age-pleiotropy or trait-pleiotropy are needed.
    A composite genetic architecture (CGA) cannot simulate pleiotropic sites, it can only simulate non-pleiotropic sites;
    while an MGA can simulate both.
    
    

    Similarly to the composite genetic architecture (CGA; class CompositeArchitecture),
    in an MGA every site affects the 
    To specify 
    Under a modifying genetic architecture, a 0-genome (a genome that is only composed of zeros)

    """

    def __init__(self, ploid, PHENOMAP, AGE_LIMIT):
        self.ploid = ploid
        self.PHENOMAP = PHENOMAP
        self.AGE_LIMIT = AGE_LIMIT

        self.gpm_decoder = GPM_decoder(PHENOMAP)

        self.length = self.gpm_decoder.n

        phenolist = self.gpm_decoder.get_total_phenolist()
        self.phenomap = GPM(
            AGE_LIMIT=AGE_LIMIT,
            phenomatrix=None,
            phenolist=phenolist,
        )

    def get_number_of_phenotypic_values(self):
        return self.AGE_LIMIT * constants.TRAIT_N

    def get_number_of_bits(self):
        return self.length * self.ploid.y

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
