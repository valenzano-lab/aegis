import logging
from aegis_sim.dataclasses.bitarray import Genomes, Origins
from aegis_sim.submodels.reproduction.mutation import Mutator
from aegis_sim.submodels.reproduction import pairing, origin_compatibility, recombination


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

    def __init__(
        self,
        RECOMBINATION_RATE,
        REPRODUCTION_MODE,
        mutator,
        ORIGIN_TRACKING,
        ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY,
    ):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.ORIGIN_TRACKING = ORIGIN_TRACKING
        self.ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY = (
            ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY
        )
        self.mutator: Mutator = mutator

    def generate_offspring_genomes(
        self, genomes, muta_prob, ages, parental_sexes, origins
    ) -> tuple[Genomes, Origins]:

        if self.REPRODUCTION_MODE == "sexual":

            males, females = pairing.get_mating_pairs(parental_sexes)
            logging.debug(f"Number of pairs after pairing: {len(males)}")

            males, females = (
                origin_compatibility.compute_position_dependent_incompatibility(
                    males=males,
                    females=females,
                    origins=origins,
                    ORIGIN_TRACKING=self.ORIGIN_TRACKING,
                    ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY=self.ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY,
                )
            )
            logging.debug(f"Number of pairs after origin incompatibility: {len(males)}")

            # TODO Make more efficient by skipping individuals which are not mating due to origin incompatibility
            if self.RECOMBINATION_RATE > 0:
                n_recombination_sites, chiasmata_list = recombination.get_recombination_parameters(
                    genomes, self.RECOMBINATION_RATE
                )
                genomes = recombination.recombination_via_pairs(
                    genomes, n_recombination_sites, chiasmata_list
                )
                if origins is not None:
                    origins = recombination.recombination_via_pairs(
                        origins, n_recombination_sites, chiasmata_list
                    )

            ages = ages[females]
            muta_prob = muta_prob[females]

            assert len(males) == len(
                females
            ), "Number of successfully reproducing males and females must be equal"
            which_male_gamete, which_female_gamete = pairing.get_which_gametes(
                n_pairs=len(males)
            )

            genomes = pairing.pair(
                bitarray=Genomes(genomes),
                males=males,
                females=females,
                which_male_gamete=which_male_gamete,
                which_female_gamete=which_female_gamete,
            )
            if origins is not None:
                origins = pairing.pair(
                    bitarray=Origins(origins),
                    males=males,
                    females=females,
                    which_male_gamete=which_male_gamete,
                    which_female_gamete=which_female_gamete,
                )

        genomes = self.mutator._mutate(genomes, muta_prob, ages)
        genomes = Genomes(genomes)
        if not isinstance(origins, Origins) and origins is not None:
            origins = Origins(origins)

        return genomes, origins
