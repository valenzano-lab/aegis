import numpy as np
import logging

from aegis.modules.reproducer import Reproducer
from aegis.modules.overshoot import Overshoot
from aegis.modules.recorder import Recorder
from aegis.modules.season import Season
from aegis.modules.gstruc import Gstruc
from aegis.modules.population import Population

from aegis.panconfiguration import pan


class Ecosystem:
    """Ecosystem

    Contains all logic and data necessary to simulate one population.
    """

    def __init__(self, id_, population=None):
        # TODO when loading from a pickle, load the envmap too

        self.id_ = id_  # Important when there are multiple populations

        logging.info("Initialized ecosystem %s", self.id_)

        # Initialize ecosystem variables
        self.max_uid = 0  # ID of the most recently born individual

        # Initialize genome structure
        self.gstruc = Gstruc(
            pan.params_list[self.id_],
            BITS_PER_LOCUS=self._get_param("BITS_PER_LOCUS"),
            REPRODUCTION_MODE=self._get_param("REPRODUCTION_MODE"),
        )  # TODO You should not pass all parameters

        # Initialize recorder
        self.recorder = Recorder(
            ecosystem_id=self.id_,
            MAX_LIFESPAN=self._get_param("MAX_LIFESPAN"),
        )

        if self.gstruc.phenomap.map_ is not None:
            self.recorder.record_phenomap(self.gstruc.phenomap.map_)

        # Initialize reproducer
        self.reproducer = Reproducer(
            RECOMBINATION_RATE=self._get_param("RECOMBINATION_RATE"),
            MUTATION_RATIO=self._get_param("MUTATION_RATIO"),
            REPRODUCTION_MODE=self._get_param("REPRODUCTION_MODE"),
        )

        # Initialize season
        self.season = Season(STAGES_PER_SEASON=self._get_param("STAGES_PER_SEASON"))

        # Initialize overshoot
        self.overshoot = Overshoot(
            OVERSHOOT_EVENT=self._get_param("OVERSHOOT_EVENT"),
            MAX_POPULATION_SIZE=self._get_param("MAX_POPULATION_SIZE"),
            CLIFF_SURVIVORSHIP=self._get_param("CLIFF_SURVIVORSHIP"),
        )

        # Initialize population
        if population is not None:
            self.population = population
        else:
            num = self._get_param("MAX_POPULATION_SIZE")
            HEADSUP = self._get_param("HEADSUP")
            headsup = (
                HEADSUP + self._get_param("MATURATION_AGE") if HEADSUP > -1 else None
            )

            genomes = self.gstruc.initialize_genomes(num, headsup)
            ages = np.zeros(num, dtype=np.int32)
            births = np.zeros(num, dtype=np.int32)
            birthdays = np.zeros(num, dtype=np.int32)
            phenotypes = self.gstruc.get_phenotype(genomes)

            self.population = Population(genomes, ages, births, birthdays, phenotypes)

    ##############
    # MAIN LOGIC #
    ##############

    def run_stage(self):
        """Perform one stage of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if self.population.n_total == 0:
            self.recorder.extinct = True
            return

        # If no living individuals
        if self.population.n_alive:
            self.eco_survival()
            self.gen_survival()
            self.reproduction()
            self.age()

        self.season_step()

        self.population.reshuffle()

        # Evolve environment if applicable
        self.gstruc.environment.evolve()

        # Population census
        self.recorder.collect("cumulative_ages", self.population.ages)

        # Record data
        self.recorder.record_pickle(self.population)
        self.recorder.record_snapshots(self.population)
        self.recorder.record_visor(self.population)
        self.recorder.record_popgenstats(
            self.population.genomes, self._get_evaluation
        )  # TODO defers calculation of mutation rates; hacky

    ###############
    # STAGE LOGIC #
    ###############

    def eco_survival(self):
        """Impose ecological death, i.e. death that arises due to resource scarcity."""
        # NOTE At this point population.alive_ only contains True
        mask_kill = self.overshoot(n=self.population.n_alive)
        self.population.mask(~mask_kill)
        self.recorder.collect("age_at_overshoot", self.population.ages[mask_kill])

    def gen_survival(self):
        """Impose genomic death, i.e. death that arises with probability encoded in the genome."""
        probs_surv = self._get_evaluation("surv", mask=self.population.alive_)
        mask_surv = pan.rng.random(len(probs_surv), dtype=np.float32) < probs_surv
        self.population.mask(mask_surv)
        self.recorder.collect("age_at_genetic", self.population.ages[~mask_surv])

    def reproduction(self):
        """Generate offspring of reproducing individuals."""

        # Check for alive mature individuals
        mask = self.population.alive_ * (
            self.population.ages >= self._get_param("MATURATION_AGE")
        )
        if not any(mask):
            return

        # Check if reproducing
        probs_repr = self._get_evaluation("repr", mask=mask)
        mask_repr = pan.rng.random(len(probs_repr), dtype=np.float32) < probs_repr

        # Forgo if not at least two available parents
        if np.count_nonzero(mask_repr) < 2:
            return

        # Count ages at reproduction
        self.recorder.collect("age_at_birth", self.population.ages[mask_repr])

        # Increase births statistics
        self.population.births += mask_repr

        # Generate offspring genomes
        parents = self.population.genomes[mask_repr]
        muta_prob = self._get_evaluation("muta", mask=mask_repr)[mask_repr]
        genomes = self.reproducer(parents, muta_prob)

        # Get eggs
        n = len(genomes)
        eggs = Population(
            genomes=genomes,
            ages=np.zeros(n, dtype=np.int32),
            births=np.zeros(n, dtype=np.int32),
            birthdays=np.zeros(n, dtype=np.int32) + pan.stage,
            phenotypes=self.gstruc.get_phenotype(genomes),
        )

        self.population.add_eggs(eggs)

    def age(self):
        """Increase age of all by one and kill those that surpass max lifespan."""
        self.population.ages += 1
        mask_kill = self.population.ages >= self._get_param("MAX_LIFESPAN")
        self.population.mask(~mask_kill)
        # NOTE Not recording or collecting ages at max_lifespan

    def season_step(self):
        """Let one time unit pass in the season.
        Kill the population if the season is over, and hatch the saved eggs."""
        self.season.countdown -= 1
        if self.season.countdown == 0:
            # Kill all living
            self.recorder.collect(
                "season_shift",
                self.population.ages[self.population.alive_],
            )
            self.population.mask(~self.population.alive_)

            # Hatch eggs and restart season
            self.population.hatch_ = True
            self.season.start_new()

        elif self.season.countdown == float("inf"):
            # Add newborns to population
            self.population.hatch_ = True

    ################
    # HELPER FUNCS #
    ################

    def _get_evaluation(self, attr, mask=None):
        """Return an array of phenotypic values for a trait.
        Use argument mask if values for some individuals are not necessary. Those will not be calculated but set to 0.
        """

        # TODO rethinking masking and indexing

        which_individuals = np.arange(self.population.n_self)
        if mask is not None:
            which_individuals = which_individuals[mask]

        # first scenario
        trait = self.gstruc[attr]
        if not trait.evolvable:
            probs = trait.initial

        # second and third scenario
        if trait.evolvable:
            which_loci = trait.start
            if trait.agespecific:
                which_loci += self.population.ages[which_individuals]

            probs = self.population.phenotypes[which_individuals, which_loci]

        # expand values back into an array with shape of whole population
        final_probs = np.zeros(self.population.n_self, dtype=np.float32)
        final_probs[which_individuals] += probs

        return final_probs

    def _get_param(self, param):
        """Get parameter value for this specific ecosystem."""
        return pan.params_list[self.id_][param]
