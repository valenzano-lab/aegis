"""
Abstract away genetic architecture.
"""

from aegis_sim.submodels.genetics.envdrift import Envdrift
from aegis_sim.submodels.genetics.composite.architecture import CompositeArchitecture
from aegis_sim.submodels.genetics.modifying.architecture import ModifyingArchitecture
from aegis_sim.dataclasses.phenotypes import Phenotypes


class Architect:
    """Wrapper for a genetic architecture"""

    def __init__(self, BITS_PER_LOCUS, PHENOMAP, AGE_LIMIT, THRESHOLD, ENVDRIFT_RATE, GENARCH_TYPE, MODIF_GENOME_SIZE):

        assert GENARCH_TYPE in ("modifying", "composite")

        if GENARCH_TYPE == "modifying":
            assert (
                BITS_PER_LOCUS == 1
            ), f"BITS_PER_LOCUS should be 1 if PHENOMAP is specified but it is set to {BITS_PER_LOCUS}"
            architecture = ModifyingArchitecture(
                PHENOMAP=PHENOMAP,
                AGE_LIMIT=AGE_LIMIT,
                MODIF_GENOME_SIZE=MODIF_GENOME_SIZE,
            )

        elif GENARCH_TYPE == "composite":
            architecture = CompositeArchitecture(
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
        assert envgenomes.shape == genomes.shape()  # envgenome retains the same shape as genome array

        pheno_array = self.architecture.compute(envgenomes)
        assert len(pheno_array) == len(genomes)  # no individuals are lost during the computation

        return Phenotypes(pheno_array)
