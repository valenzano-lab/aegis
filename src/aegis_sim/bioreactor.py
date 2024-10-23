import numpy as np
import logging

from aegis_sim import variables
from aegis_sim import submodels
from aegis_sim.constants import VALID_CAUSES_OF_DEATH
from aegis_sim.dataclasses.population import Population
from aegis_sim.recording import recordingmanager
from aegis_sim.parameterization import parametermanager


class Bioreactor:
    def __init__(self, population: Population):
        self.eggs: Population = None
        self.population: Population = population

    ##############
    # MAIN LOGIC #
    ##############

    def run_step(self):
        """Perform one step of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if len(self) == 0:
            logging.debug("Population went extinct.")
            recordingmanager.summaryrecorder.extinct = True
        # Mortality sources
        self.mortalities()

        recordingmanager.popsizerecorder.write_before_reproduction(self.population)
        self.growth()  # size increase
        self.reproduction()  # reproduction
        self.age()  # age increment and potentially death
        self.hatch()
        submodels.architect.envdrift.evolve(step=variables.steps)
        submodels.resources.replenish()

        # Record data
        recordingmanager.popsizerecorder.write_after_reproduction(self.population)
        recordingmanager.flushrecorder.collect("additive_age_structure", self.population.ages)  # population census
        recordingmanager.picklerecorder.write(self.population)
        recordingmanager.featherrecorder.write(self.population)
        recordingmanager.guirecorder.record(self.population)
        recordingmanager.flushrecorder.flush()
        recordingmanager.popgenstatsrecorder.write(
            self.population.genomes, self.population.phenotypes.extract(ages=self.population.ages, trait_name="muta")
        )  # TODO defers calculation of mutation rates; hacky
        recordingmanager.summaryrecorder.record_memuse()
        recordingmanager.terecorder.record(self.population.ages, "alive")

    ###############
    # STEP LOGIC #
    ###############

    def mortalities(self):
        for source in parametermanager.parameters.MORTALITY_ORDER:
            if source == "intrinsic":
                self.mortality_intrinsic()
            elif source == "abiotic":
                self.mortality_abiotic()
            elif source == "infection":
                self.mortality_infection()
            elif source == "predation":
                self.mortality_predation()
            elif source == "starvation":
                self.mortality_starvation()
            else:
                raise ValueError(f"Invalid source of mortality '{source}'")

    def mortality_intrinsic(self):
        probs_surv = self.population.phenotypes.extract(ages=self.population.ages, trait_name="surv")
        age_hazard = submodels.frailty.modify(hazard=1 - probs_surv, ages=self.population.ages)
        mask_kill = np.random.random(len(probs_surv)) < age_hazard
        self._kill(mask_kill=mask_kill, causeofdeath="intrinsic")

    def mortality_abiotic(self):
        hazard = submodels.abiotic(variables.steps)
        age_hazard = submodels.frailty.modify(hazard=hazard, ages=self.population.ages)
        mask_kill = np.random.random(len(self.population)) < age_hazard
        self._kill(mask_kill=mask_kill, causeofdeath="abiotic")

    def mortality_infection(self):
        submodels.infection(self.population)
        # TODO add age hazard
        mask_kill = self.population.infection == -1
        self._kill(mask_kill=mask_kill, causeofdeath="infection")

    def mortality_predation(self):
        probs_kill = submodels.predation(len(self))
        # TODO add age hazard
        mask_kill = np.random.random(len(self)) < probs_kill
        self._kill(mask_kill=mask_kill, causeofdeath="predation")

    def mortality_starvation(self):
        resource_availability = submodels.resources.scavenge(np.ones(len(self.population)))
        # TODO add age hazard
        mask_kill = submodels.starvation(
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
            self.population.ages >= parametermanager.parameters.MATURATION_AGE
        )  # Check if mature; mature if survived MATURATION_AGE full cycles
        if parametermanager.parameters.REPRODUCTION_ENDPOINT > 0:
            mask_menopausal = (
                self.population.ages >= parametermanager.parameters.REPRODUCTION_ENDPOINT
            )  # Check if menopausal; menopausal when lived through REPRODUCTION_ENDPOINT full cycles
            mask_fertile = (mask_fertile) & (~mask_menopausal)

        if not any(mask_fertile):
            return

        # Check if reproducing
        probs_repr = self.population.phenotypes.extract(ages=self.population.ages, trait_name="repr", part=mask_fertile)

        # Binomial calculation
        n = parametermanager.parameters.MAX_OFFSPRING_NUMBER
        p = probs_repr
        num_repr = np.random.binomial(n=n, p=p)
        mask_repr = num_repr > 0

        if sum(num_repr) == 0:
            return

        # Indices of reproducing individuals
        who = np.repeat(np.arange(len(self.population)), num_repr)

        # Count ages at reproduction
        ages_repr = self.population.ages[who]
        recordingmanager.flushrecorder.collect("age_at_birth", ages_repr)

        # Increase births statistics
        self.population.births += num_repr

        # Generate offspring genomes
        parental_genomes = self.population.genomes.get(individuals=who)
        parental_sexes = self.population.sexes[who]

        muta_prob = self.population.phenotypes.extract(ages=self.population.ages, trait_name="muta", part=mask_repr)[
            mask_repr
        ]
        muta_prob = np.repeat(muta_prob, num_repr[mask_repr])

        offspring_genomes = submodels.reproduction.generate_offspring_genomes(
            genomes=parental_genomes,
            muta_prob=muta_prob,
            ages=ages_repr,
            parental_sexes=parental_sexes,
        )
        offspring_sexes = submodels.sexsystem.get_sex(len(offspring_genomes))

        # Randomize order of newly laid egg attributes ..
        # .. because the order will affect their probability to be removed because of limited carrying capacity
        order = np.arange(len(offspring_sexes))
        np.random.shuffle(order)
        offspring_genomes = offspring_genomes[order]
        offspring_sexes = offspring_sexes[order]

        # Make eggs
        eggs = Population.make_eggs(
            offspring_genomes=offspring_genomes,
            step=variables.steps,
            offspring_sexes=offspring_sexes,
            parental_generations=np.zeros(len(offspring_sexes)),  # TODO replace with working calculation
        )
        if self.eggs is None:
            self.eggs = eggs
        else:
            self.eggs += eggs
        if len(self.eggs) > parametermanager.parameters.CARRYING_CAPACITY_EGGS:
            indices = np.arange(len(self.eggs))[-parametermanager.parameters.CARRYING_CAPACITY_EGGS :]
            # TODO biased
            self.eggs *= indices

    def growth(self):
        # TODO use already scavenged resources to determine growth
        # max_growth_potential = self.population.phenotypes.extract(ages=self.population.ages, trait_name="grow")
        # gathered_resources = submodels.resources.scavenge(max_growth_potential)
        # self.population.sizes += gathered_resources
        self.population.sizes += 1

    def age(self):
        """Increase age of all by one and kill those that surpass age limit.
        Age denotes the number of full cycles that an individual survived and reproduced.
        AGE_LIMIT is the maximum number of full cycles an individual can go through.
        """
        self.population.ages += 1
        mask_kill = self.population.ages >= parametermanager.parameters.AGE_LIMIT
        self._kill(mask_kill=mask_kill, causeofdeath="age_limit")

    def hatch(self):
        """Turn eggs into living individuals"""

        # If nothing to hatch
        if self.eggs is None:
            return

        # If something to hatch
        if (
            (
                parametermanager.parameters.INCUBATION_PERIOD == -1 and len(self.population) == 0
            )  # hatch when everyone dead
            or (parametermanager.parameters.INCUBATION_PERIOD == 0)  # hatch immediately
            or (
                parametermanager.parameters.INCUBATION_PERIOD > 0
                and variables.steps % parametermanager.parameters.INCUBATION_PERIOD == 0
            )  # hatch with delay
        ):
            self.eggs.phenotypes = submodels.architect.__call__(self.eggs.genomes)
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
        # if causeofdeath != "age_limit":
        ages_death = self.population.ages[mask_kill]
        recordingmanager.flushrecorder.collect(f"age_at_{causeofdeath}", ages_death)
        recordingmanager.terecorder.record(ages_death, "dead")

        # Retain survivors
        self.population *= ~mask_kill

    def __len__(self):
        """Return the number of living individuals and saved eggs."""
        return len(self.population) + len(self.eggs) if self.eggs is not None else len(self.population)
