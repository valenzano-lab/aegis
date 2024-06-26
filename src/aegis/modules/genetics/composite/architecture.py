import numpy as np
from aegis.hermes import hermes
from aegis import constants

from aegis.modules.genetics.composite.interpreter import Interpreter


class CompositeArchitecture:
    """

    VISOR
    - when pleiotropy is not needed;
    - it is quick, easy to analyze, delivers a diversity of phenotypes
    - every trait (surv repr muta neut) can be evolvable or not
    - if not evolvable, the value is set by !!!
    - if evolvable, it can be agespecific or age-independent
    - probability of a trait at each age is determined by a BITS_PER_LOCUS adjacent bits forming a "locus" / gene
    - the method by which these loci are converted into a phenotypic value is the Interpreter type

    
    ### COMPOSITE GENETIC ARCHITECTURE ###
    In AEGIS, genetic architecture refers to the structure of the pseudogenome.

    Every pseudogenome is a threedimensional bitstring (/ 3D array of bits).
    It is a bitstring because it consists 0's and 1's with a fixed order.
    It is threedimensional because the position of every site is expressed using three numbers:
    `chromosome set`, `locus index`, `bit index`


    ### COMPUTATION ###

    Translation of pseudogenome into a phenotype requires two steps.
    First is the application of inheritance patterns (done by class Ploid), the second is the locus computation.


    phenotype(age, trait) = sum_{i=0}^{BITS_PER_LOCUS}(haplogenotype[age, i] * weight(i))

    Under a composite genetic architecture, 
    
    Under a composite genetic architecture, the pseudogenome has three dimensions;
    i.e. to reference a certain site, we need to specify three numbers:
    the chromosome set index, locus index and sublocus index.
    
    The chromosome set index exists because the individuals might be diploid or polyploid,
    so they might carry two or more chromosome sets.

    The locus index 

    """

    def __init__(self, ploid, BITS_PER_LOCUS, AGE_LIMIT, THRESHOLD):
        self.ploid = ploid
        self.BITS_PER_LOCUS = BITS_PER_LOCUS
        self.n_loci = constants.TRAIT_N * AGE_LIMIT
        self.length = self.n_loci * BITS_PER_LOCUS
        self.AGE_LIMIT = AGE_LIMIT

        self.evolvable = [trait for trait in hermes.traits.values() if trait.evolvable]

        self.interpreter = Interpreter(
            self.BITS_PER_LOCUS,
            THRESHOLD,
        )

    def get_number_of_phenotypic_values(self):
        return self.AGE_LIMIT * constants.TRAIT_N

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
