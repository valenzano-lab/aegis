"""
Abstract away genetic architecture.
"""

import numpy as np

from aegis_sim.submodels.genetics.envdrift import Envdrift
from aegis_sim.submodels.genetics.composite.architecture import CompositeArchitecture
from aegis_sim.submodels.genetics.modifying.architecture import ModifyingArchitecture
from aegis_sim import parameterization

from aegis_sim.dataclasses.phenotypes import Phenotypes


class Architect:
    """Wrapper for a genetic architecture"""

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
        for trait in parameterization.traits.values():
            phenotypes[:, trait.slice] = Architect.clip(phenotypes[:, trait.slice], trait.name)

        return Phenotypes(phenotypes)

    @staticmethod
    def clip(array, traitname):
        lo = parameterization.traits[traitname].lo
        hi = parameterization.traits[traitname].hi
        return lo + array * (hi - lo)
