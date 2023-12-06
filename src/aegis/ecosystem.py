import numpy as np

from aegis import cnf
from aegis import pan

from aegis.help.config import causeofdeath_valid
from aegis.modules import recorder
from aegis.modules.genetics import gstruc, phenomap, flipmap, phenotyper
from aegis.modules.population import Population
from aegis.modules.mortality import disease, environment, overshoot, predation
from aegis.modules.reproduction import recombination, assortment, mutation


class Ecosystem:
    """Ecosystem

    Contains all logic and data necessary to simulate one population.
    """

    def __init__(self, population=None):
        # TODO when loading from a pickle, load the envmap too

        # Initialize ecosystem variables
        self.max_uid = 0  # ID of the most recently born individual

        # Initialize recorder

        if phenomap.map_ is not None:
            recorder.record_phenomap(phenomap.map_)

        # Initialize eggs
        self.eggs = None

        # Initialize population
        if population is not None:
            self.population = population
        else:
            genomes = gstruc.initialize_genomes()
            ages = np.zeros(cnf.MAX_POPULATION_SIZE, dtype=np.int32)
            births = np.zeros(cnf.MAX_POPULATION_SIZE, dtype=np.int32)
            birthdays = np.zeros(cnf.MAX_POPULATION_SIZE, dtype=np.int32)
            phenotypes = phenotyper.get(genomes)
            disease_ = np.zeros(cnf.MAX_POPULATION_SIZE, dtype=np.int32)

            self.population = Population(genomes, ages, births, birthdays, phenotypes, disease_)

    ##############
    # MAIN LOGIC #
    ##############

    def run_stage(self):
        """Perform one stage of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if len(self) == 0:
            recorder.extinct = True
            return

        # If no living individuals
        if len(self.population):
            self.eco_survival()
            self.gen_survival()
            self.env_survival()
            self.dis_survival()
            self.pred_survival()
            self.reproduction()
            self.age()

        self.season_step()

        # Evolve flipmap if applicable
        flipmap.evolve()

        # Population census
        recorder.collect("additive_age_structure", self.population.ages)

        # Record data
        recorder.record_pickle(self.population)
        recorder.record_snapshots(self.population)
        recorder.record_visor(self.population)
        recorder.record_popgenstats(
            self.population.genomes, self._get_evaluation
        )  # TODO defers calculation of mutation rates; hacky

        # Memory use
        recorder.record_memory_use()

    ###############
    # STAGE LOGIC #
    ###############

    def age(self):
        """Increase age of all by one and kill those that surpass max lifespan."""
        self.population.ages += 1
        mask_kill = self.population.ages >= cnf.MAX_LIFESPAN
        self._kill(mask_kill=mask_kill, causeofdeath="max_lifespan")

    def env_survival(self):
        """Impose environmental hazard death; i.e. death due to abiotic and cyclical factors such as temperature."""
        hazard = environment.get_hazard(pan.stage)
        mask_kill = pan.rng.random(len(self.population), dtype=np.float32) < hazard
        self._kill(mask_kill=mask_kill, causeofdeath="environment")

    def eco_survival(self):
        """Impose ecological death, i.e. death that arises due to resource scarcity."""
        mask_kill = overshoot.call(n=len(self.population))
        self._kill(mask_kill=mask_kill, causeofdeath="overshoot")

    def gen_survival(self):
        """Impose genomic death, i.e. death that arises with probability encoded in the genome."""
        probs_surv = self._get_evaluation("surv")
        mask_surv = pan.rng.random(len(probs_surv), dtype=np.float32) < probs_surv
        self._kill(mask_kill=~mask_surv, causeofdeath="genetic")

    def dis_survival(self):
        """Impose death due to infection."""
        disease.kill(self.population)
        mask_kill = self.population.disease == -1
        self._kill(mask_kill=mask_kill, causeofdeath="disease")

    def pred_survival(self):
        """Impose predation death"""
        probs_kill = predation.call(len(self))
        mask_kill = pan.rng.random(len(self), dtype=np.float32) < probs_kill
        self._kill(mask_kill=mask_kill, causeofdeath="predation")

    def season_step(self):
        """Let one time unit pass in the season.
        Kill the population if the season is over, and hatch the saved eggs."""
        pan.season_countdown -= 1
        if pan.season_countdown == 0:
            # Kill all living
            mask_kill = np.ones(len(self.population), dtype=np.bool_)
            self._kill(mask_kill, "season_shift")

            # Hatch eggs and restart season
            self._hatch_eggs()
            pan.season_countdown += cnf.STAGES_PER_SEASON

        elif pan.season_countdown == float("inf"):
            # Add newborns to population
            self._hatch_eggs()

    def reproduction(self):
        """Generate offspring of reproducing individuals."""

        # Check if fertile
        mask_fertile = self.population.ages > cnf.MATURATION_AGE  # Check if mature
        if cnf.MENOPAUSE > 0:
            mask_menopausal = self.population.ages >= cnf.MENOPAUSE  # Check if menopausal
            mask_fertile = (mask_fertile) & (~mask_menopausal)
        if not any(mask_fertile):
            return

        # Check if reproducing
        probs_repr = self._get_evaluation("repr", part=mask_fertile)
        mask_repr = pan.rng.random(len(probs_repr), dtype=np.float32) < probs_repr

        # Forgo if not at least two available parents
        if np.count_nonzero(mask_repr) < 2:
            return

        # Count ages at reproduction
        ages_repr = self.population.ages[mask_repr]
        recorder.collect("age_at_birth", ages_repr)

        # Increase births statistics
        self.population.births += mask_repr

        # Generate offspring genomes
        genomes = self.population.genomes[mask_repr]  # parental genomes
        muta_prob = self._get_evaluation("muta", part=mask_repr)[mask_repr]

        if cnf.REPRODUCTION_MODE == "sexual":
            genomes = recombination.do(genomes)
            genomes, _ = assortment.do(genomes)

        genomes = mutation._mutate(genomes, muta_prob)

        # Get eggs
        n = len(genomes)
        eggs = Population(
            genomes=genomes,
            ages=np.zeros(n, dtype=np.int32),
            births=np.zeros(n, dtype=np.int32),
            birthdays=np.zeros(n, dtype=np.int32) + pan.stage,
            phenotypes=phenotyper.get(genomes),
            disease=np.zeros(n, dtype=np.int32),
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
        trait = gstruc.traits[attr]
        if not trait.evolvable:
            probs = trait.initial

        # second and third scenario
        if trait.evolvable:
            which_loci = trait.start
            if trait.agespecific:
                which_loci += self.population.ages[which_individuals]

            probs = self.population.phenotypes[which_individuals, which_loci]

        # expand values back into an array with shape of whole population
        final_probs = np.zeros(len(self.population), dtype=np.float32)
        final_probs[which_individuals] += probs

        return final_probs

    def _kill(self, mask_kill, causeofdeath):
        """Kill individuals and record their data.
        Killing can occur due to age, genomic death, ecological death, and season shift.
        """

        assert causeofdeath in causeofdeath_valid

        # Skip if no one to kill
        if not any(mask_kill):
            return

        # Count ages at death
        if causeofdeath != "max_lifespan":
            ages_death = self.population.ages[mask_kill]
            recorder.collect(f"age_at_{causeofdeath}", ages_death)

        # Retain survivors
        self.population *= ~mask_kill

    def __len__(self):
        """Return the number of living individuals and saved eggs."""
        return len(self.population) + len(self.eggs) if self.eggs is not None else len(self.population)
