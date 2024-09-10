"""
Abstract away genetic architecture.
"""

import numpy as np

from aegis_sim.hermes import hermes

from aegis_sim.modules.genetics.envdrift import Envdrift
from aegis_sim.modules.genetics.composite.architecture import CompositeArchitecture
from aegis_sim.modules.genetics.modifying.architecture import ModifyingArchitecture


class Architect:
    """Wrapper for a genetic architecture

    ### INTRINSIC PHENOTYPE VS TOTAL/REALIZED/OBSERVED PHENOTYPE ###
    AEGIS distinguishes between a total/realized/observed phenotype and an intrinsic phenotype.

    The observed phenotype refers to the realized survival and reproduction events that depend both on
    the properties of the individual itself, its interaction with other individuals (e.g. as mates)
    and the environment (e.g. resources, predators).
    The intrinsic phenotype refers to the properties of the individual itself – it is the phenotype that
    would be observed if there are no external limitations (e.g. no environmental limitations through
    limited resources, or no reproductive limitations through low availability of mates).

    The observed phenotype, therefore, is what is commonly referred to as simply 'phenotype', while
    the intrinsic phenotype captures the empirically elusive concept of the hidden biological capacities
    of an individual in an idealized environment.
    The intrinsic phenotype is directly and solely shaped by genetics, while the observed phenotype
    depends on extrinsic factors as well.

    ### GENETIC ARCHITECTURE: DEFINITION ###
    In AEGIS, every individual carries a genome which encodes an intrinsic phenotype.
    Genetic architecture refers to that encoding; i.e. it describes the structure of a genome
    enabling us to translate a genotype into a phenotype.

    ### GENETIC ARCHITECTURE: COMPOSITE VS MODIFYING ###
    In AEGIS, we distinguish between two kinds of genetic architectures:
    a composite genetic architecture (CGA) and a modifying genetic architecture (MGA).
    
    The main advantage of the modifying genetic architecture (MGA) is its ability to simulate pleiotropic and non-pleiotropic variants
    (while the composite architecture can only simulate non-pleiotropic variants).

    The main advantage of the composite genetic architecture (CGA) is that it results in faster simulations, it is easier to set up
    (from the user's perspective), and easier to later analyze and visualize. Also, while under the MGA, inadvertent epistasis can occur
    (it might happen that a certain site has no effect on the phenotype), while that never happens under the CGA.


    ### GENETIC ARCHITECTURE: REAL GENOMES VS PSEUDOGENOMES ###
    Pseudogenomes are similar and different to real genomes in multiple ways.

    Both are composed of a fixed but different number of bases (four bases for real: A, T, G, C; two bases for pseudogenomes: 0 and 1).
    For both, we can define a primary structure (e.g. AGGCTTACTA for real genomes; e.g. 0111010110 for pseudogenomes), which can be specified as a simple array.
    Furthermore, in both, a variant at any 
    
    While real genomes are composed of four bases (A, T, G, and C), pseudogenomes are composed of two (0 and 1).
    For 

    Real genomes have multiple structural levels – 
    Real genomes 

    Pseudogenomes are multidimensional bitstrings – structured arrays of 0's and 1's.
    Similarly
    
    Refer to the example below:

    EXAMPLE:

    An unraveled pseudogenome: 0100101000001000
   
               set 1      set 2
    locus 1    01001010   10110101
    locus 2    00001000   10100101
    locus 3    00010010   00010010
    ...       
    locus L-2  11011111   11011000
    locus L-1  11001000   10110010
    locus L    00000001   01110101

    In the schematic



    """

    def __init__(
        self,
        BITS_PER_LOCUS,
        PHENOMAP,
        AGE_LIMIT,
        THRESHOLD,
        ploid,
        ENVDRIFT_RATE,
    ):

        if PHENOMAP:
            assert (
                BITS_PER_LOCUS == 1
            ), f"BITS_PER_LOCUS should be 1 if PHENOMAP is specified but it is set to {BITS_PER_LOCUS}"
            architecture = ModifyingArchitecture(
                ploid=ploid,
                PHENOMAP=PHENOMAP,
                AGE_LIMIT=AGE_LIMIT,
            )

        else:
            architecture = CompositeArchitecture(
                ploid=ploid,
                BITS_PER_LOCUS=BITS_PER_LOCUS,
                AGE_LIMIT=AGE_LIMIT,
                THRESHOLD=THRESHOLD,
            )
        self.architecture = architecture
        self.envdrift = Envdrift(ENVDRIFT_RATE=ENVDRIFT_RATE, genome_shape=self.architecture.get_shape())

    def __call__(self, genomes):
        """Translate genomes into an array of phenotypes probabilities."""

        # Apply the envdrift
        envgenomes = self.envdrift.call(genomes.get_array())

        phenotypes = self.architecture.compute(envgenomes)

        # Apply lo and hi bound
        # TODO extract slicing
        for trait in hermes.traits.values():
            phenotypes[:, trait.slice] = Architect.clip(phenotypes[:, trait.slice], trait.name)

        return phenotypes

    @staticmethod
    def clip(array, traitname):
        lo = hermes.traits[traitname].lo
        hi = hermes.traits[traitname].hi
        return lo + array * (hi - lo)

    def get_evaluation(self, population, attr, part=None):
        """
        Get phenotypic values of a certain trait for certain individuals.
        Note that the function returns 0 for individuals that are to not be evaluated.
        """

        # TODO Shift some responsibilities to Phenotypes dataclass

        which_individuals = np.arange(len(population))
        if part is not None:
            which_individuals = which_individuals[part]

        # first scenario
        trait = hermes.traits[attr]
        if not trait.evolvable:
            probs = trait.initpheno

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
