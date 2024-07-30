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

    def run_step(self):
        """Perform one step of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if len(self) == 0:
            logging.debug("Population went extinct.")
            hermes.recording_manager.summaryrecorder.extinct = True

        # Mortality sources
        self.mortality_intrinsic()
        self.mortality_abiotic()
        self.mortality_infection()
        self.mortality_predation()
        self.mortality_starvation()

        self.growth()  # size increase
        self.reproduction()  # reproduction
        self.age()  # age increment and potentially death
        self.hatch()
        hermes.architect.envdrift.evolve(step=hermes.get_step())
        hermes.modules.resources.replenish()

        # Record data
        hermes.recording_manager.popsizerecorder.write(self.population)
        hermes.recording_manager.flushrecorder.collect("additive_age_structure", self.population.ages)  # population census
        hermes.recording_manager.picklerecorder.write(self.population)
        hermes.recording_manager.featherrecorder.write(self.population)
        hermes.recording_manager.guirecorder.record(self.population)
        hermes.recording_manager.flushrecorder.flush()
        hermes.recording_manager.popgenstatsrecorder.write(
            self.population.genomes, hermes.architect.get_evaluation(self.population, "muta")
        )  # TODO defers calculation of mutation rates; hacky
        hermes.recording_manager.summaryrecorder.record_memuse()
        hermes.recording_manager.terecorder.record(self.population.ages, "alive")

    ###############
    # STEP LOGIC #
    ###############

    def mortality_intrinsic(self):
        probs_surv = hermes.architect.get_evaluation(self.population, "surv")
        mask_surv = hermes.rng.random(len(probs_surv), dtype=np.float32) < probs_surv
        self._kill(mask_kill=~mask_surv, causeofdeath="intrinsic")

    def mortality_abiotic(self):
        hazard = hermes.modules.abiotic(hermes.get_step())
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
        resource_availability = hermes.modules.resources.scavenge(np.ones(len(self.population)))
        mask_kill = hermes.modules.starvation(
            n=len(self.population),
            resource_availability=resource_availability.sum(),
        )
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
        probs_repr = hermes.architect.get_evaluation(self.population, "repr", part=mask_fertile)
        mask_repr = hermes.rng.random(len(probs_repr), dtype=np.float32) < probs_repr

        # Forgo if not at least two available parents
        if np.count_nonzero(mask_repr) < 2:
            return

        # Count ages at reproduction
        ages_repr = self.population.ages[mask_repr]
        hermes.recording_manager.flushrecorder.collect("age_at_birth", ages_repr)

        # Increase births statistics
        self.population.births += mask_repr

        # Generate offspring genomes
        parental_genomes = self.population.genomes.get(individuals=mask_repr)
        parental_sexes = self.population.sexes[mask_repr]
        muta_prob = hermes.architect.get_evaluation(self.population, "muta", part=mask_repr)[mask_repr]
        offspring_genomes = hermes.modules.reproduction.generate_offspring_genomes(
            genomes=parental_genomes,
            muta_prob=muta_prob,
            ages=ages_repr,
            parental_sexes=parental_sexes,
        )
        offspring_sexes = hermes.modules.sexsystem.get_sex(len(offspring_genomes))

        # Get eggs
        eggs = Population.make_eggs(
            offspring_genomes=offspring_genomes, step=hermes.get_step(), offspring_sexes=offspring_sexes
        )
        if self.eggs is None:
            self.eggs = eggs
        else:
            self.eggs += eggs

    def growth(self):
        max_growth_potential = hermes.architect.get_evaluation(self.population, "grow")
        gathered_resources = hermes.modules.resources.scavenge(max_growth_potential)
        self.population.sizes += gathered_resources

    def age(self):
        """Increase age of all by one and kill those that surpass age limit.
        Age denotes the number of full cycles that an individual survived and reproduced.
        AGE_LIMIT is the maximum number of full cycles an individual can go through.
        """
        self.population.ages += 1
        mask_kill = self.population.ages >= hermes.parameters.AGE_LIMIT
        self._kill(mask_kill=mask_kill, causeofdeath="age_limit")

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
                and hermes.get_step() % hermes.parameters.INCUBATION_PERIOD == 0
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
        if causeofdeath != "age_limit":
            ages_death = self.population.ages[mask_kill]
            hermes.recording_manager.flushrecorder.collect(f"age_at_{causeofdeath}", ages_death)
            hermes.recording_manager.terecorder.record(ages_death, "dead")

        # Retain survivors
        self.population *= ~mask_kill

    def __len__(self):
        """Return the number of living individuals and saved eggs."""
        return len(self.population) + len(self.eggs) if self.eggs is not None else len(self.population)
