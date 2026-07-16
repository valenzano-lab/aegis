import numpy as np
from aegis_sim import constants
from aegis_sim import variables

from aegis_sim.submodels.genetics.composite.interpreter import Interpreter
from aegis_sim import parameterization
from aegis_sim.parameterization import parametermanager
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

        # Optional pleiotropy. None unless PHENOMAP_SPECS is given.
        self.phenomap_matrix = self._build_phenomap(parametermanager.parameters.PHENOMAP_SPECS)

    def _build_phenomap(self, PHENOMAP_SPECS):
        """Build a genotype-phenotype matrix from PHENOMAP_SPECS (aegis v1 semantics).

        The matrix is the identity plus off-diagonal weights: the diagonal means every
        locus keeps encoding its own trait at its own age (so age-specific survival is
        preserved), and each spec adds a *pleiotropic* effect of one locus on another
        trait/age on top. This is what makes antagonistic pleiotropy expressible --
        one locus raising surv at an early age and lowering it at a late one.

        Each spec is ``[source_trait, source_index, target_trait, target_age, weight]``
        with ``source_index`` (1..trait length) and ``target_age`` (1..AGE_LIMIT) 1-based,
        as exported by v1. Indices are resolved in *logical* (trait x age) order, which is
        the order compute() reorders genomes into.

        Returns None when no specs are given, in which case compute() is unchanged.
        """
        if not PHENOMAP_SPECS:
            return None

        map_ = np.diag(np.ones(self.n_loci, dtype=np.float32))
        for spec in PHENOMAP_SPECS:
            source_trait, source_index, target_trait, target_age, weight = spec
            source = parameterization.traits[source_trait]
            target = parameterization.traits[target_trait]
            geno_i = source.start + (int(source_index) - 1)
            pheno_i = target.start + (int(target_age) - 1)
            assert source.start <= geno_i < source.end, (
                f"PHENOMAP_SPECS source index {source_index} is out of range for trait "
                f"'{source_trait}' which has {source.length} loci; "
                f"set G_{source_trait}_agespecific to at least {source_index}"
            )
            assert target.start <= pheno_i < target.end, (
                f"PHENOMAP_SPECS target age {target_age} is out of range for trait "
                f"'{target_trait}' which has {target.length} loci"
            )
            map_[geno_i, pheno_i] = np.float32(weight)
        return map_

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
            interpretome[:, trait.slice] += probs  # add back

        # Apply pleiotropy, if any, to the raw [0, 1] interpreter output -- before the
        # lo/hi mapping below, so that spec weights are expressed in interpreter units
        # (aegis v1 order: interpret -> phenomap -> lo/hi).
        if self.phenomap_matrix is not None:
            interpretome = np.clip(interpretome.dot(self.phenomap_matrix), 0, 1).astype(np.float32)

        # Map the [0, 1] interpreter output onto the trait's [lo, hi] phenotypic range.
        # This is what makes G_<trait>_lo / G_<trait>_hi do anything — without this scaling
        # the lo/hi parameters are parsed but discarded. Defaults: G_surv_lo=0.7, G_surv_hi=1.0
        # (surv never goes to 0); G_repr_lo=0, G_repr_hi=0.5; others lo=0, hi=1.
        for trait in parameterization.traits.values():
            interpretome[:, trait.slice] = trait.lo + (trait.hi - trait.lo) * interpretome[:, trait.slice]

        return interpretome

    # def diffuse(self, probs):
    #     window_size = parametermanager.parameters.DIFFUSION_FACTOR * 2 + 1
    #     p = np.empty(shape=(probs.shape[0], probs.shape[1] + window_size - 1))
    #     p[:, :window_size] = np.repeat(probs[:, 0], window_size).reshape(-1, window_size)
    #     p[:, window_size - 1 :] = probs[:]
    #     diffusome = np.convolve(p[0], np.ones(window_size) / window_size, mode="valid")

    def get_map(self):
        pass
