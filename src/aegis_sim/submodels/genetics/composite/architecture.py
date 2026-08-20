import numpy as np
from aegis_sim import constants
from aegis_sim import variables

from aegis_sim.submodels.genetics.composite.interpreter import Interpreter
from aegis_sim import parameterization
from aegis_sim.submodels.genetics import ploider


class CompositeArchitecture:
    """

    GUI
    - when pleiotropy is not needed;
    - it is quick, easy to analyze, delivers a diversity of phenotypes
    - every trait (surv repr muta neut) can be evolvable or not
    - if not evolvable, the value is set by !!!
    - if evolvable, it can be agespecific or age-independent
    - probability of a trait at each age is determined by a BITS_PER_LOCUS adjacent bits forming a "locus" / gene
    - the method by which these loci are converted into a phenotypic value is the Interpreter type

    """

    def __init__(self, BITS_PER_LOCUS, AGE_LIMIT, THRESHOLD):
        self.BITS_PER_LOCUS = BITS_PER_LOCUS
        self.n_loci = sum(trait.length for trait in parameterization.traits.values())
        self.length = self.n_loci * BITS_PER_LOCUS
        self.AGE_LIMIT = AGE_LIMIT

        self.evolvable = [trait for trait in parameterization.traits.values() if trait.evolvable]

        self.interpreter = Interpreter(
            self.BITS_PER_LOCUS,
            THRESHOLD,
        )

        # Fixed seed=0 so all populations (including hybridizing ones with different
        # RANDOM_SEEDs) share an identical physical genome layout.
        # bit_permutation[i] = physical bit position of logical bit i, where "logical"
        # means the (trait, age, bit-within-locus) order that the rest of AEGIS
        # (Trait.slice, the Interpreter, recorders, ...) assumes. Storage (genomes,
        # mutation, recombination) always operates in *physical* order; recombination
        # is a linear scan over physical bit position, so two loci -- or even two bits
        # of the same locus -- that happen to sit next to each other logically (e.g.
        # consecutive ages of the same trait, or adjacent bits of one locus) are no
        # longer more likely to be co-inherited than any other pair of bits. This
        # generalizes the locus-level permutation to individual bits, so linkage no
        # longer correlates with logical genome layout at all, not even within a locus.
        self.bit_permutation = np.random.default_rng(0).permutation(self.length)

    def get_number_of_bits(self):
        return ploider.ploider.y * self.n_loci * self.BITS_PER_LOCUS

    def get_shape(self):
        return (ploider.ploider.y, self.n_loci, self.BITS_PER_LOCUS)

    def to_logical(self, array):
        """Reorder an array from physical (storage) bit order into logical (trait x age x bit) order.

        Works on any array whose last two axes are (n_loci, BITS_PER_LOCUS) --
        e.g. (popsize, n_loci, BITS_PER_LOCUS) or (popsize, ploidy, n_loci, BITS_PER_LOCUS).
        """
        leading_shape = array.shape[:-2]
        flat = array.reshape(*leading_shape, self.length)
        flat = flat[..., self.bit_permutation]
        return flat.reshape(*leading_shape, self.n_loci, self.BITS_PER_LOCUS)

    def to_physical(self, array):
        """Reorder an array from logical (trait x age x bit) order into physical (storage) bit order.

        Inverse of to_logical.
        """
        leading_shape = array.shape[:-2]
        flat = array.reshape(*leading_shape, self.length)
        physical = np.empty_like(flat)
        physical[..., self.bit_permutation] = flat
        return physical.reshape(*leading_shape, self.n_loci, self.BITS_PER_LOCUS)

    def init_genome_array(self, popsize):
        # TODO enable agespecific False

        # Fill uniformly at random first -- this is i.i.d. per bit so physical vs.
        # logical order does not matter here. Only the per-trait threshold below has
        # to land on the correct *physical* bits.
        array = variables.rng.random(size=(popsize, *self.get_shape()))
        flat = array.reshape(popsize, ploider.ploider.y, self.length)

        for trait in parameterization.traits.values():
            logical_bits = slice(trait.start * self.BITS_PER_LOCUS, trait.end * self.BITS_PER_LOCUS)
            phys_pos = self.bit_permutation[logical_bits]
            flat[:, :, phys_pos] = flat[:, :, phys_pos] < trait.initgeno

        return array

    def init_origins_array(self, popsize, origin_tracking_number):
        return np.ones(shape=(popsize, *self.get_shape())) * origin_tracking_number

    def compute(self, genomes):

        if genomes.shape[1] == 1:  # Do not calculate mean if genomes are haploid
            genomes = genomes[:, 0]
        else:
            genomes = ploider.ploider.diploid_to_haploid(genomes)

        # Reorder from physical storage order to logical (trait x age x bit) order
        genomes = self.to_logical(genomes)

        interpretome = np.zeros(shape=(genomes.shape[0], genomes.shape[1]), dtype=np.float32)
        for trait in parameterization.traits.values():
            loci = genomes[:, trait.slice]  # fetch
            probs = self.interpreter.call(loci, trait.interpreter)  # interpret
            # self.diffuse(probs)
            interpretome[:, trait.slice] += probs  # add back

        return interpretome

    # def diffuse(self, probs):
    #     window_size = parametermanager.parameters.DIFFUSION_FACTOR * 2 + 1
    #     p = np.empty(shape=(probs.shape[0], probs.shape[1] + window_size - 1))
    #     p[:, :window_size] = np.repeat(probs[:, 0], window_size).reshape(-1, window_size)
    #     p[:, window_size - 1 :] = probs[:]
    #     diffusome = np.convolve(p[0], np.ones(window_size) / window_size, mode="valid")

    def get_map(self):
        pass
