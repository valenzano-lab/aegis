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

        self.id_ = id_  # Important when there are multiple populations

        logging.info("Initialized ecosystem %s", self.id_)

        # Initialize ecosystem variables
        self.max_uid = 0  # ID of the most recently born individual

        # Initialize recorder
        self.recorder = Recorder(
            ecosystem_id=self.id_,
            MAX_LIFESPAN=self._get_param("MAX_LIFESPAN"),
        )

        # Initialize genome structure
        self.gstruc = Gstruc(
            pan.params_list[self.id_],
            BITS_PER_LOCUS=self._get_param("BITS_PER_LOCUS"),
            REPRODUCTION_MODE=self._get_param("REPRODUCTION_MODE"),
        )  # TODO You should not pass all parameters

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

        # Initialize eggs
        self.eggs = None

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
            ages = np.zeros(num, int)
            births = np.zeros(num, int)
            birthdays = np.zeros(num, int)
            phenotypes = self.gstruc.get_phenotype(genomes)

            self.population = Population(genomes, ages, births, birthdays, phenotypes)

    ##############
    # MAIN LOGIC #
    ##############

    def run_stage(self):
        """Perform one stage of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if len(self) == 0:
            self.recorder.extinct = True
            return

        self.recorder.record_snapshots(self.population)
        self.recorder.record_visor(self.population)
        self.recorder.record_popgenstats(self.population)
        self.recorder.collect("cumulative_ages", self.population.ages)

        if len(self.population):  # no living individuals
            self.eco_survival()
            self.gen_survival()
            self.reproduction()
            self.age()

        self.season_step()

        # Evolve environment if applicable
        self.gstruc.environment.evolve()

    ###############
    # STAGE LOGIC #
    ###############

    def age(self):
        """Increase age of all by one and kill those that surpass max lifespan."""
        self.population.ages += 1
        mask_kill = self.population.ages >= self._get_param("MAX_LIFESPAN")
        self._kill(mask_kill=mask_kill, causeofdeath="max_lifespan")

    def eco_survival(self):
        """Impose ecological death, i.e. death that arises due to resource scarcity."""
        mask_kill = self.overshoot(n=len(self.population))
        self._kill(mask_kill=mask_kill, causeofdeath="overshoot")

    def gen_survival(self):
        """Impose genomic death, i.e. death that arises with probability encoded in the genome."""
        probs_surv = self._get_evaluation("surv")
        mask_surv = pan.rng.random(len(probs_surv)) < probs_surv
        self._kill(mask_kill=~mask_surv, causeofdeath="genetic")

    def season_step(self):
        """Let one time unit pass in the season.
        Kill the population if the season is over, and hatch the saved eggs."""
        self.season.countdown -= 1
        if self.season.countdown == 0:
            # Kill all living
            mask_kill = np.ones(len(self.population), bool)
            self._kill(mask_kill, "season_shift")

            # Hatch eggs and restart season
            self._hatch_eggs()
            self.season.start_new()

        elif self.season.countdown == float("inf"):
            # Add newborns to population
            self._hatch_eggs()

    def reproduction(self):
        """Generate offspring of reproducing individuals."""

        # Check if mature
        mask_mature = self.population.ages >= self._get_param("MATURATION_AGE")
        if not any(mask_mature):
            return

        # Check if reproducing
        probs_repr = self._get_evaluation("repr", part=mask_mature)
        mask_repr = pan.rng.random(len(probs_repr)) < probs_repr
        if sum(mask_repr) < 2:  # Forgo if not at least two available parents
            return

        # Count ages at reproduction
        ages_repr = self.population.ages[mask_repr]
        self.recorder.collect("age_at_birth", ages_repr)

        # Increase births statistics
        self.population.births += mask_repr

        # Generate offspring genomes
        parents = self.population.genomes[mask_repr]
        muta_prob = self._get_evaluation("muta", part=mask_repr)[mask_repr]
        genomes = self.reproducer(parents, muta_prob)

        # Get eggs
        n = len(genomes)
        eggs = Population(
            genomes=genomes,
            ages=np.zeros(n, int),
            births=np.zeros(n, int),
            birthdays=np.zeros(n, int) + pan.stage,
            phenotypes=self.gstruc.get_phenotype(genomes),
        )

        if self.eggs is None:
            self.eggs = eggs
        else:
            self.eggs += eggs

    ################
    # HELPER FUNCS #
    ################

    def _hatch_eggs(self):
        """Add offspring from eggs into the living population."""
        if self.eggs is not None:
            self.population += self.eggs
            self.eggs = None

    def _get_evaluation(self, attr, part=None):
        """Get phenotypic values of a certain trait for a certain individuals."""
        which_individuals = np.arange(len(self.population))
        if part is not None:
            which_individuals = which_individuals[part]

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
        final_probs = np.zeros(len(self.population))
        final_probs[which_individuals] += probs

        return final_probs

    def _kill(self, mask_kill, causeofdeath):
        """Kill individuals and record their data.
        Killing can occur due to age, genomic death, ecological death, and season shift.
        """

        # Skip if no one to kill
        if not any(mask_kill):
            return

        # Count ages at death
        if causeofdeath != "max_lifespan":
            ages_death = self.population.ages[mask_kill]
            # self.macroconfig.counter.count(f"age_at_{causeofdeath}", ages_death)
            self.recorder.collect(f"age_at_{causeofdeath}", ages_death)

        # Retain survivors
        self.population *= ~mask_kill

    def __len__(self):
        """Return the number of living individuals and saved eggs."""
        return (
            len(self.population) + len(self.eggs)
            if self.eggs is not None
            else len(self.population)
        )

    def _get_param(self, param):
        """Get parameter value for this specific ecosystem."""
        return pan.params_list[self.id_][param]
