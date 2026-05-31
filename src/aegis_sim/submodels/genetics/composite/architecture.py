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

        # Fixed seed=0 so all populations (including hybridizing ones with different RANDOM_SEEDs)
        # share an identical physical genome layout; locus_permutation[i] = physical position of logical locus i
        self.locus_permutation = np.random.default_rng(0).permutation(self.n_loci)

        # Per-locus dominance coefficient h, indexed by *physical* locus position
        # (because diploid_to_haploid operates on the physical layout, before reorder).
        # Built from each trait's G_<trait>_dominance value (default 0.5 = codominant).
        self.dominance_per_locus = np.full(self.n_loci, 0.5, dtype=np.float32)
        for trait in parameterization.traits.values():
            if trait.length == 0:
                continue
            phys_pos = self.locus_permutation[trait.start:trait.end]
            self.dominance_per_locus[phys_pos] = np.float32(trait.dominance)

    def get_number_of_bits(self):
        return ploider.ploider.y * self.n_loci * self.BITS_PER_LOCUS

    def get_shape(self):
        return (ploider.ploider.y, self.n_loci, self.BITS_PER_LOCUS)

    def init_genome_array(self, popsize):
        # TODO enable agespecific False
        array = variables.rng.random(size=(popsize, *self.get_shape()))

        for trait in parameterization.traits.values():
            phys_pos = self.locus_permutation[trait.start:trait.end]
            array[:, :, phys_pos, :] = array[:, :, phys_pos, :] < trait.initgeno

        return array

    def compute(self, genomes):

        if genomes.shape[1] == 1:  # Do not calculate mean if genomes are haploid
            genomes = genomes[:, 0]
        else:
            genomes = ploider.ploider.diploid_to_haploid(genomes, dominance_per_locus=self.dominance_per_locus)

        # Reorder from physical storage order to logical (trait×age) order
        genomes = genomes[:, self.locus_permutation, :]

        interpretome = np.zeros(shape=(genomes.shape[0], genomes.shape[1]), dtype=np.float32)
        for trait in parameterization.traits.values():
            loci = genomes[:, trait.slice]  # fetch
            probs = self.interpreter.call(loci, trait.interpreter)  # interpret
            # Map the [0, 1] interpreter output onto the trait's [lo, hi] phenotypic range.
            # This is what makes G_<trait>_lo / G_<trait>_hi do anything — without this scaling
            # the lo/hi parameters are parsed but discarded. Defaults: G_surv_lo=0.7, G_surv_hi=1.0
            # (surv never goes to 0); G_repr_lo=0, G_repr_hi=0.5; others lo=0, hi=1.
            probs = trait.lo + (trait.hi - trait.lo) * probs
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
