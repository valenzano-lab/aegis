import numpy as np

from aegis_sim import constants
from aegis_sim import parameterization

from aegis_sim.submodels.genetics.modifying.gpm_decoder import GPM_decoder
from aegis_sim.submodels.genetics.modifying.gpm import GPM
from aegis_sim.submodels.genetics import ploider


class ModifyingArchitecture:
    """

    GUI
    - when pleiotropy is needed
    - when all bits are 0, the phenotypic values are the ones set from parameters (baseline set in parameters);
    vs composite where it would be 0.
    - ... dev still required
    """

    def __init__(self, PHENOMAP, AGE_LIMIT, MODIF_GENOME_SIZE):
        self.PHENOMAP = PHENOMAP

        self.gpm_decoder = GPM_decoder(PHENOMAP)

        self.length = MODIF_GENOME_SIZE

        phenolist = self.gpm_decoder.get_total_phenolist()
        self.phenomap = GPM(
            phenomatrix=None,
            phenolist=phenolist,
        )

        self.n_phenotypic_values = AGE_LIMIT * constants.TRAIT_N

        self.AGE_LIMIT = AGE_LIMIT

    def get_number_of_bits(self):
        return self.length * ploider.ploider.y

    def get_shape(self):
        return (ploider.ploider.y, self.length, 1)

    def init_genome_array(self, popsize):
        return np.zeros(shape=(popsize, *self.get_shape()), dtype=np.bool_)

    def init_phenotype_array(self, popsize):
        return np.zeros(shape=(popsize, self.n_phenotypic_values))

    def compute(self, genomes):

        if genomes.shape[1] == 1:  # Do not calculate mean if genomes are haploid
            genomes = genomes[:, 0]
        else:
            genomes = ploider.ploider.diploid_to_haploid(genomes)

        # TODO yuck!

        # Apply phenomap
        phenomapped = self.phenomap(
            interpretome=genomes.reshape(len(genomes), -1),
            zeropheno=self.init_phenotype_array(
                popsize=len(genomes),
            ),
        )

        # TODO damn ugly!

        # Add background values
        for traitname, trait in parameterization.traits.items():
            start = constants.starting_site(trait.name) * self.AGE_LIMIT
            end = start + self.AGE_LIMIT
            phenomapped[:, slice(start, end)] += trait.initpheno

            # Check that phenotype values are within [0,1]:
            p = phenomapped[:, slice(start, end)]
            p[p > 1] = 1
            p[p < 0] = 0
            phenomapped[:, slice(start, end)] = p

        return phenomapped
