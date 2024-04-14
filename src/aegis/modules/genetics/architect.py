"""
Abstract away genetic architecture.
"""

import numpy as np

from aegis.hermes import hermes

from aegis.modules.genetics.envdrift import Envdrift
from aegis.modules.genetics.composite.architecture import CompositeArchitecture
from aegis.modules.genetics.modifying.architecture import ModifyingArchitecture


class Architect:
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
