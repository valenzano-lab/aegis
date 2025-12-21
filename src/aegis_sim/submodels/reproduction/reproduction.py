import logging
from aegis_sim.dataclasses.bitarray import BitArray, Genomes, Origins
from aegis_sim.submodels.reproduction.mutation import Mutator
from aegis_sim.submodels.reproduction.pairing import pairing, get_mating_pairs
from aegis_sim.submodels.reproduction.recombination import (
    recombination,
    recombination_via_pairs,
    get_recombination_parameters,
)


class Reproducer:
    """
    GUI
    Individuals are fertile starting with [[MATURATION_AGE]] (can be 0) until [[REPRODUCTION_ENDPOINT]] (if 0, no REPRODUCTION_ENDPOINT occurs).
    Reproduction can be sexual (with diploid genomes) or asexual (with diploid or haploid genomes).
    When reproduction is sexual, recombination occurs in gametes at a rate of [[RECOMBINATION_RATE]]
    and gametes will inherit mutations at an age-independent rate
    which can be parameterized (genetics-independent) or set to evolve (genetics-dependent).
    Mutations cause the offspring genome bit states to flip from 0-to-1 or 1-to-0.
    The ratio of 0-to-1 and 1-to-0 can be modified using the [[MUTATION_RATIO]].

    If the population is oviparous, [[INCUBATION_PERIOD]] should be set to -1, 1 or greater.
    When it is set to -1, all laid eggs hatch only once all living individuals die.
    When it is set to 0 or greater, eggs hatch after that specified time.
    Thus, when 0, individuals do not go through an egg stage during their life cycle.
    """

    # TODO INCUBATION_PERIOD set to -1 or 1 or greater is stupid
    # TODO INCUBATION_PERIOD is not really a part of this submodel, but it is in the documentation. it should be though.

    # TODO probably better to split mutation logic into another domain and cluster together with genetic architecture stuff

    def __init__(self, RECOMBINATION_RATE, REPRODUCTION_MODE, mutator):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.mutator: Mutator = mutator

    def generate_offspring_genomes(
        self, genomes, muta_prob, ages, parental_sexes, origins
    ) -> tuple[Genomes, Origins]:

        if self.REPRODUCTION_MODE == "sexual":
            # genomes = recombination(genomes, self.RECOMBINATION_RATE)
            if self.RECOMBINATION_RATE > 0:
                n_recombination_sites, chiasmata_list = get_recombination_parameters(
                    genomes, self.RECOMBINATION_RATE
                )
                genomes = recombination_via_pairs(
                    genomes, n_recombination_sites, chiasmata_list
                )
                if origins is not None:
                    origins = recombination_via_pairs(
                        origins, n_recombination_sites, chiasmata_list
                    )

            males, females, n_pairs, which_male_gamete, which_female_gamete = (
                get_mating_pairs(parental_sexes)
            )
            ages = ages[females]
            muta_prob = muta_prob[females]

            genomes = pairing(
                Genomes(genomes),
                males,
                females,
                n_pairs,
                which_male_gamete=which_male_gamete,
                which_female_gamete=which_female_gamete,
            )
            if origins is not None:
                origins = pairing(
                    Origins(origins),
                    males,
                    females,
                    n_pairs,
                    which_male_gamete=which_male_gamete,
                    which_female_gamete=which_female_gamete,
                )

        genomes = self.mutator._mutate(genomes, muta_prob, ages)
        genomes = Genomes(genomes)
        if not isinstance(origins, Origins) and origins is not None:
            origins = Origins(origins)

        return genomes, origins
