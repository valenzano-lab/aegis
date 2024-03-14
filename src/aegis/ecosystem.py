import numpy as np
import logging

from aegis.pan import cnf, rng, get_stage

from aegis.help.config import causeofdeath_valid
from aegis.modules import recorder
from aegis.modules.population import Population
from aegis.modules.mortality import abiotic, infection, predation, starvation
from aegis.modules import genetics


class Ecosystem:
    """Ecosystem

    Contains all logic and data necessary to simulate one population.
    """

    def __init__(self, population=None):
        # TODO when loading from a pickle, load the envmap too

        # Initialize eggs
        self.eggs = None

        # Initialize population
        if population is not None:
            self.population = population
        else:
            self.population = Population.initialize(N=cnf.MAX_POPULATION_SIZE)

    ##############
    # MAIN LOGIC #
    ##############

    def run_stage(self):
        """Perform one stage of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if len(self) == 0:
            logging.debug("went extinct")
            recorder.extinct = True

        # Mortality sources
        self.mortality_intrinsic()
        self.mortality_abiotic()
        self.mortality_infection()
        self.mortality_predation()
        self.mortality_starvation()

        self.reproduction()  # reproduction
        self.age()  # age increment and potentially death
        self.hatch()
        genetics.flipmap_evolve(stage=get_stage())

        # Record data
        recorder.collect("additive_age_structure", self.population.ages)  # population census
        recorder.record_pickle(self.population)
        recorder.record_snapshots(self.population)
        recorder.record_visor(self.population)
        recorder.record_popgenstats(
            self.population.genomes, genetics.get_evaluation(self.population, "muta")
        )  # TODO defers calculation of mutation rates; hacky
        recorder.record_memory_use()
        recorder.record_TE(self.population.ages, "alive")

    ###############
    # STAGE LOGIC #
    ###############

    def mortality_intrinsic(self):
        probs_surv = genetics.get_evaluation(self.population, "surv")
        mask_surv = rng.random(len(probs_surv), dtype=np.float32) < probs_surv
        self._kill(mask_kill=~mask_surv, causeofdeath="intrinsic")

    def mortality_abiotic(self):
        hazard = abiotic.get_hazard(get_stage())
        mask_kill = rng.random(len(self.population), dtype=np.float32) < hazard
        self._kill(mask_kill=mask_kill, causeofdeath="abiotic")

    def mortality_infection(self):
        infection.kill(self.population)
        mask_kill = self.population.infection == -1
        self._kill(mask_kill=mask_kill, causeofdeath="infection")

    def mortality_predation(self):
        probs_kill = predation.call(len(self))
        mask_kill = rng.random(len(self), dtype=np.float32) < probs_kill
        self._kill(mask_kill=mask_kill, causeofdeath="predation")

    def mortality_starvation(self):
        mask_kill = starvation.call(n=len(self.population))
        self._kill(mask_kill=mask_kill, causeofdeath="starvation")

    def reproduction(self):
        """Generate offspring of reproducing individuals.
        Initial is set to 0.
        """

        # Check if fertile
        mask_fertile = (
            self.population.ages >= cnf.MATURATION_AGE
        )  # Check if mature; mature if survived MATURATION_AGE full cycles
        if cnf.MENOPAUSE > 0:
            mask_menopausal = (
                self.population.ages >= cnf.MENOPAUSE
            )  # Check if menopausal; menopausal when lived through MENOPAUSE full cycles
            mask_fertile = (mask_fertile) & (~mask_menopausal)
        if not any(mask_fertile):
            return

        # Check if reproducing
        probs_repr = genetics.get_evaluation(self.population, "repr", part=mask_fertile)
        mask_repr = rng.random(len(probs_repr), dtype=np.float32) < probs_repr

        # Forgo if not at least two available parents
        if np.count_nonzero(mask_repr) < 2:
            return

        # Count ages at reproduction
        ages_repr = self.population.ages[mask_repr]
        recorder.collect("age_at_birth", ages_repr)

        # Increase births statistics
        self.population.births += mask_repr

        # Generate offspring genomes
        parental_genomes = self.population.genomes.get(individuals=mask_repr)  # parental genomes
        muta_prob = genetics.get_evaluation(self.population, "muta", part=mask_repr)[mask_repr]
        offspring_genomes = self.population.genomes.generate_offspring_genomes(
            genomes=parental_genomes, muta_prob=muta_prob
        )

        # Get eggs
        eggs = Population.make_eggs(offspring_genomes=offspring_genomes, stage=get_stage())
        if self.eggs is None:
            self.eggs = eggs
        else:
            self.eggs += eggs

    def age(self):
        """Increase age of all by one and kill those that surpass max lifespan.
        Age denotes the number of full cycles that an individual survived and reproduced.
        MAX_LIFESPAN is the maximum number of full cycles an individual can go through.
        """
        self.population.ages += 1
        mask_kill = self.population.ages >= cnf.MAX_LIFESPAN
        self._kill(mask_kill=mask_kill, causeofdeath="max_lifespan")

    def hatch(self):
        """Turn eggs into living individuals"""

        # If nothing to hatch
        if self.eggs is None:
            return

        # If something to hatch
        if (
            (cnf.INCUBATION_PERIOD == -1 and len(self.population) == 0)  # hatch when everyone dead
            or (cnf.INCUBATION_PERIOD == 0)  # hatch immediately
            or (cnf.INCUBATION_PERIOD > 0 and get_stage() % cnf.INCUBATION_PERIOD == 0)  # hatch with delay
        ):
            self.population += self.eggs
            self.eggs = None

    ################
    # HELPER FUNCS #
    ################

    def _kill(self, mask_kill, causeofdeath):
        """Kill individuals and record their data."""

        assert causeofdeath in causeofdeath_valid

        # Skip if no one to kill
        if not any(mask_kill):
            return

        # Count ages at death
        if causeofdeath != "max_lifespan":
            ages_death = self.population.ages[mask_kill]
            recorder.collect(f"age_at_{causeofdeath}", ages_death)
            recorder.record_TE(ages_death, "dead")

        # Retain survivors
        self.population *= ~mask_kill

    def __len__(self):
        """Return the number of living individuals and saved eggs."""
        return len(self.population) + len(self.eggs) if self.eggs is not None else len(self.population)
