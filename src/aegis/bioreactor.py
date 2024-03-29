import numpy as np
import logging

from aegis.constants import VALID_CAUSES_OF_DEATH
from aegis.modules.dataclasses.population import Population
from aegis.hermes import hermes


class Bioreactor:
    def __init__(self, population):
        self.eggs = None
        self.population = population

    ##############
    # MAIN LOGIC #
    ##############

    def run_stage(self):
        """Perform one stage of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if len(self) == 0:
            logging.debug("went extinct")
            hermes.recorder.summaryrecorder.extinct = True

        # Mortality sources
        self.mortality_intrinsic()
        self.mortality_abiotic()
        self.mortality_infection()
        self.mortality_predation()
        self.mortality_starvation()

        self.reproduction()  # reproduction
        self.age()  # age increment and potentially death
        self.hatch()
        hermes.modules.architect.flipmap.evolve(stage=hermes.get_stage())

        # Record data
        hermes.recorder.flushrecorder.collect("additive_age_structure", self.population.ages)  # population census
        hermes.recorder.picklerecorder.write(self.population)
        hermes.recorder.featherrecorder.write(self.population)
        hermes.recorder.visorrecorder.record(self.population)
        hermes.recorder.flushrecorder.flush()
        hermes.recorder.popgenstatsrecorder.write(
            self.population.genomes, hermes.modules.architect.get_evaluation(self.population, "muta")
        )  # TODO defers calculation of mutation rates; hacky
        hermes.recorder.summaryrecorder.record_memuse()
        hermes.recorder.terecorder.record(self.population.ages, "alive")

    ###############
    # STAGE LOGIC #
    ###############

    def mortality_intrinsic(self):
        probs_surv = hermes.modules.architect.get_evaluation(self.population, "surv")
        mask_surv = hermes.rng.random(len(probs_surv), dtype=np.float32) < probs_surv
        self._kill(mask_kill=~mask_surv, causeofdeath="intrinsic")

    def mortality_abiotic(self):
        hazard = hermes.modules.abiotic(hermes.get_stage())
        mask_kill = hermes.rng.random(len(self.population), dtype=np.float32) < hazard
        self._kill(mask_kill=mask_kill, causeofdeath="abiotic")

    def mortality_infection(self):
        hermes.modules.infection(self.population)
        mask_kill = self.population.infection == -1
        self._kill(mask_kill=mask_kill, causeofdeath="infection")

    def mortality_predation(self):
        probs_kill = hermes.modules.predation(len(self))
        mask_kill = hermes.rng.random(len(self), dtype=np.float32) < probs_kill
        self._kill(mask_kill=mask_kill, causeofdeath="predation")

    def mortality_starvation(self):
        mask_kill = hermes.modules.starvation(n=len(self.population))
        self._kill(mask_kill=mask_kill, causeofdeath="starvation")

    def reproduction(self):
        """Generate offspring of reproducing individuals.
        Initial is set to 0.
        """

        # Check if fertile
        mask_fertile = (
            self.population.ages >= hermes.parameters.MATURATION_AGE
        )  # Check if mature; mature if survived MATURATION_AGE full cycles
        if hermes.parameters.MENOPAUSE > 0:
            mask_menopausal = (
                self.population.ages >= hermes.parameters.MENOPAUSE
            )  # Check if menopausal; menopausal when lived through MENOPAUSE full cycles
            mask_fertile = (mask_fertile) & (~mask_menopausal)
        if not any(mask_fertile):
            return

        # Check if reproducing
        probs_repr = hermes.modules.architect.get_evaluation(self.population, "repr", part=mask_fertile)
        mask_repr = hermes.rng.random(len(probs_repr), dtype=np.float32) < probs_repr

        # Forgo if not at least two available parents
        if np.count_nonzero(mask_repr) < 2:
            return

        # Count ages at reproduction
        ages_repr = self.population.ages[mask_repr]
        hermes.recorder.flushrecorder.collect("age_at_birth", ages_repr)

        # Increase births statistics
        self.population.births += mask_repr

        # Generate offspring genomes
        parental_genomes = self.population.genomes.get(individuals=mask_repr)  # parental genomes
        muta_prob = hermes.modules.architect.get_evaluation(self.population, "muta", part=mask_repr)[mask_repr]
        offspring_genomes = hermes.modules.reproduction.generate_offspring_genomes(
            genomes=parental_genomes, muta_prob=muta_prob
        )

        # Get eggs
        eggs = Population.make_eggs(offspring_genomes=offspring_genomes, stage=hermes.get_stage())
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
        mask_kill = self.population.ages >= hermes.parameters.MAX_LIFESPAN
        self._kill(mask_kill=mask_kill, causeofdeath="max_lifespan")

    def hatch(self):
        """Turn eggs into living individuals"""

        # If nothing to hatch
        if self.eggs is None:
            return

        # If something to hatch
        if (
            (hermes.parameters.INCUBATION_PERIOD == -1 and len(self.population) == 0)  # hatch when everyone dead
            or (hermes.parameters.INCUBATION_PERIOD == 0)  # hatch immediately
            or (
                hermes.parameters.INCUBATION_PERIOD > 0
                and hermes.get_stage() % hermes.parameters.INCUBATION_PERIOD == 0
            )  # hatch with delay
        ):
            self.population += self.eggs
            self.eggs = None

    ################
    # HELPER FUNCS #
    ################

    def _kill(self, mask_kill, causeofdeath):
        """Kill individuals and record their data."""

        assert causeofdeath in VALID_CAUSES_OF_DEATH

        # Skip if no one to kill
        if not any(mask_kill):
            return

        # Count ages at death
        if causeofdeath != "max_lifespan":
            ages_death = self.population.ages[mask_kill]
            hermes.recorder.flushrecorder.collect(f"age_at_{causeofdeath}", ages_death)
            hermes.recorder.terecorder.record(ages_death, "dead")

        # Retain survivors
        self.population *= ~mask_kill

    def __len__(self):
        """Return the number of living individuals and saved eggs."""
        return len(self.population) + len(self.eggs) if self.eggs is not None else len(self.population)
