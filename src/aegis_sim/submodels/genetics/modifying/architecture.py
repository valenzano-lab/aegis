import numpy as np

from aegis_sim import constants
from aegis_sim import variables
from aegis_sim import parameterization

from aegis_sim.parameterization import parametermanager
from aegis_sim.submodels.genetics.modifying.gpm_decoder import GPM_decoder
from aegis_sim.submodels.genetics.modifying.gpm import GPM
from aegis_sim.submodels.genetics import ploider
from aegis_sim.dataclasses.phenotypes import Phenotypes


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

        # Two ways to specify the genotype-phenotype map for the modifying architecture:
        #   1. PHENOMAP (dict shorthand) -> gpm_decoder *stochastically generates* a phenolist
        #      from block specs like {"AP, 88": [["surv", "agespec", 0.003]]}.
        #   2. PHENOMAP_SPECS (explicit v1 phenolist) -> a flat list of
        #      [source, site, trait, age, magnitude] quintuples that fully determine the map.
        #      This is the format exported by aegis v1 (e.g. the Ne/MA/AP paper configs).
        # If PHENOMAP_SPECS is provided, use it verbatim so exact v1 maps reproduce bit-for-bit;
        # otherwise fall back to the dict-shorthand decoder.
        PHENOMAP_SPECS = parametermanager.parameters.PHENOMAP_SPECS
        if PHENOMAP_SPECS:
            phenolist = self._phenolist_from_specs(PHENOMAP_SPECS, AGE_LIMIT, MODIF_GENOME_SIZE)
        else:
            phenolist = self.gpm_decoder.get_total_phenolist()
        self.phenomap = GPM(
            phenomatrix=None,
            phenolist=phenolist,
        )

        # self.n_phenotypic_values = AGE_LIMIT * constants.TRAIT_N

        self.AGE_LIMIT = AGE_LIMIT

    @staticmethod
    def _phenolist_from_specs(PHENOMAP_SPECS, AGE_LIMIT, MODIF_GENOME_SIZE):
        """Convert the explicit v1 PHENOMAP_SPECS into a GPM phenolist.

        Each spec is a quintuple ``[source, site, trait, age, magnitude]`` where
        ``site`` (1..MODIF_GENOME_SIZE) and ``age`` (1..AGE_LIMIT) are 1-based, as
        exported by aegis v1. The GPM phenolist wants 0-based 4-tuples
        ``(vec_index, trait, age, magnitude)``, so site/age are shifted by -1.

        This is theory-agnostic: AP maps (paired +early / -late effects on the same
        site) and MA maps (single deleterious late-acting effects) are both just
        collections of such quintuples and convert identically.
        """
        phenolist = []
        for spec in PHENOMAP_SPECS:
            _source, site, trait, age, magnitude = spec
            vec_index = int(site) - 1
            age0 = int(age) - 1
            assert 0 <= vec_index < MODIF_GENOME_SIZE, (
                f"PHENOMAP_SPECS site {site} out of range for MODIF_GENOME_SIZE={MODIF_GENOME_SIZE}"
            )
            assert 0 <= age0 < AGE_LIMIT, (
                f"PHENOMAP_SPECS age {age} out of range for AGE_LIMIT={AGE_LIMIT}"
            )
            phenolist.append([vec_index, trait, age0, float(magnitude)])
        return phenolist

    def get_number_of_bits(self):
        return self.length * ploider.ploider.y

    def get_shape(self):
        return (ploider.ploider.y, self.length, 1)

    def init_genome_array(self, popsize):
        array = variables.rng.random(size=(popsize, *self.get_shape()))

        # Only neut (G_neut_initgeno) matters here
        for trait in parameterization.traits.values():
            array[:, :, trait.slice] = array[:, :, trait.slice] < trait.initgeno

        return array

    # def init_phenotype_array(self, popsize):
    #     return np.zeros(shape=(popsize, self.n_phenotypic_values))

    def compute(self, genomes):

        if genomes.shape[1] == 1:  # Do not calculate mean if genomes are haploid
            genomes = genomes[:, 0]
        else:
            genomes = ploider.ploider.diploid_to_haploid(genomes)

        # TODO yuck!

        # Apply phenomap
        phenomapped = self.phenomap(
            interpretome=genomes.reshape(len(genomes), -1),
            zeropheno=Phenotypes.init_phenotype_array(popsize=len(genomes)).array,
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
