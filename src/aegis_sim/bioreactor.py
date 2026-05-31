import numpy as np
import logging

from aegis_sim import variables
from aegis_sim import submodels
from aegis_sim.constants import VALID_CAUSES_OF_DEATH
from aegis_sim.dataclasses.population import Population
from aegis_sim.recording import recordingmanager
from aegis_sim.parameterization import parametermanager
from aegis_sim.submodels.resources.resources import resources


class Bioreactor:
    def __init__(self, population: Population):
        self.eggs: Population = None
        self.population: Population = population
        self._starvation_steps: int = 0        # consecutive steps where N > resources
        self._starvation_multiplier: float = 1.0  # (1 - STARVATION_PENALTY) ** _starvation_steps

    ##############
    # MAIN LOGIC #
    ##############

    def run_step(self):
        """Perform one step of simulation."""

        # If extinct (no living individuals nor eggs left), do nothing
        if len(self) == 0:
            logging.debug("Population went extinct.")
            recordingmanager.summaryrecorder.extinct = True
            return

        # Scavenge resources and update the starvation multiplier.
        # If N > resources: starvation counter increments and multiplier compounds.
        # If resources >= N: counter resets to 0 and multiplier returns to 1.0.
        self._scavenge_resources()

        # Mortality sources
        self.mortalities()
        resources.replenish()

        recordingmanager.popsizerecorder.write_before_reproduction(self.population)
        self.growth()  # size increase
        self.reproduction()  # reproduction
        self.age()  # age increment and potentially death
        self.hatch()
        submodels.architect.envdrift.evolve(step=variables.steps)

        # Record data
        recordingmanager.popsizerecorder.write_after_reproduction(self.population)
        recordingmanager.popsizerecorder.write_egg_num_after_reproduction(self.eggs)
        recordingmanager.envdriftmaprecorder.write(step=variables.steps)
        recordingmanager.flushrecorder.collect("additive_age_structure", self.population.ages)  # population census
        recordingmanager.picklerecorder.write(self.population)
        recordingmanager.featherrecorder.write(self.population)
        recordingmanager.ancestryrecorder.write(self.population)
        recordingmanager.fastarecorder.write(self.population)
        recordingmanager.vcfrecorder.write(self.population)
        recordingmanager.guirecorder.record(self.population)
        recordingmanager.flushrecorder.flush()
        recordingmanager.popgenstatsrecorder.write(
            self.population.genomes, self.population.phenotypes.extract(ages=self.population.ages, trait_name="muta")
        )  # TODO defers calculation of mutation rates; hacky
        recordingmanager.summaryrecorder.record_memuse()
        recordingmanager.terecorder.record(self.population.ages, "alive")
        recordingmanager.checkpointrecorder.write(self.population, self.eggs)

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
        effective_surv = probs_surv * self._starvation_multiplier
        age_hazard = submodels.frailty.modify(hazard=1 - effective_surv, ages=self.population.ages)
        mask_kill = variables.rng.random(len(probs_surv)) < age_hazard
        self._kill(mask_kill=mask_kill, causeofdeath="intrinsic")

    def mortality_abiotic(self):
        hazard = submodels.abiotic(variables.steps)
        age_hazard = submodels.frailty.modify(hazard=hazard, ages=self.population.ages)
        mask_kill = variables.rng.random(len(self.population)) < age_hazard
        self._kill(mask_kill=mask_kill, causeofdeath="abiotic")

    def mortality_infection(self):
        submodels.infection(self.population)
        # TODO add age hazard
        mask_kill = self.population.infection == -1
        self._kill(mask_kill=mask_kill, causeofdeath="infection")

    def mortality_predation(self):
        probs_kill = submodels.predation(len(self))
        # TODO add age hazard
        mask_kill = variables.rng.random(len(self)) < probs_kill
        self._kill(mask_kill=mask_kill, causeofdeath="predation")

    def mortality_starvation(self):
        # Starvation now acts by scaling surv and repr phenotypes via _resource_ratio
        # (set at the top of run_step before mortalities). No separate kill step needed.
        # This method is kept so "starvation" remains a valid MORTALITY_ORDER entry.
        pass

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

        probs_repr = (
            self.population.phenotypes.extract(ages=self.population.ages, trait_name="repr", part=mask_fertile)
            * self._starvation_multiplier
        )

        # Binomial calculation
        n = parametermanager.parameters.MAX_OFFSPRING_NUMBER
        p = probs_repr

        assert np.all(p <= 1)
        assert np.all(p >= 0)
        num_repr = variables.rng.binomial(n=n, p=p)
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
        parental_ancestry = self.population.ancestry[who] if self.population.ancestry is not None else None

        muta_prob = self.population.phenotypes.extract(ages=self.population.ages, trait_name="muta", part=mask_repr)[
            mask_repr
        ]
        muta_prob = np.repeat(muta_prob, num_repr[mask_repr])

        offspring_genomes, offspring_ancestry = submodels.reproduction.generate_offspring_genomes(
            genomes=parental_genomes,
            muta_prob=muta_prob,
            ages=ages_repr,
            parental_sexes=parental_sexes,
            ancestry=parental_ancestry,
        )
        offspring_sexes = submodels.sexsystem.get_sex(len(offspring_genomes))

        # Randomize order of newly laid egg attributes ..
        # .. because the order will affect their probability to be removed because of limited carrying capacity
        order = np.arange(len(offspring_sexes))
        variables.rng.shuffle(order)
        offspring_genomes = offspring_genomes[order]
        offspring_sexes = offspring_sexes[order]
        if offspring_ancestry is not None:
            offspring_ancestry = offspring_ancestry[order]

        # Make eggs
        eggs = Population.make_eggs(
            offspring_genomes=offspring_genomes,
            step=variables.steps,
            offspring_sexes=offspring_sexes,
            parental_generations=np.zeros(len(offspring_sexes)),  # TODO replace with working calculation
            offspring_ancestry=offspring_ancestry,
        )
        if self.eggs is None:
            self.eggs = eggs
        else:
            self.eggs += eggs

        if parametermanager.parameters.CARRYING_CAPACITY_EGGS is not None and len(self.eggs) > parametermanager.parameters.CARRYING_CAPACITY_EGGS:
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
        if self.eggs is None or len(self.eggs) == 0:
            return

        # If REPRODUCTION_REGULATION is True, only reproduce until MAX_POPULATION_SIZE
        if parametermanager.parameters.REPRODUCTION_REGULATION:
            current_population_size = len(self.population)
            remaining_capacity = resources.capacity - current_population_size
            # If no remaining capacity, do not reproduce
            if remaining_capacity < 1:
                self.eggs = None
                return
            elif remaining_capacity < len(self.eggs):
                indices = variables.rng.choice(len(self.eggs), size=int(remaining_capacity), replace=False)
                self.eggs *= indices

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

    def _scavenge_resources(self):
        """Scavenge resources and update the compounding starvation multiplier.

        If N > available resources (deficit): starvation counter increments by 1
        and multiplier = (1 - STARVATION_PENALTY) ** counter.

        If resources >= N: counter resets to 0 and multiplier returns to 1.0.

        The multiplier is applied to each individual's age-specific surv and repr
        phenotypes in mortality_intrinsic() and reproduction().
        """
        n = len(self.population)
        if n == 0:
            self._starvation_steps = 0
            self._starvation_multiplier = 1.0
            return

        in_deficit = n > resources.capacity

        recordingmanager.resourcerecorder.write_before_scavenging()
        resources.scavenge(np.ones(n))
        recordingmanager.resourcerecorder.write_after_scavenging()

        if in_deficit:
            self._starvation_steps += 1
        else:
            self._starvation_steps = 0

        penalty = parametermanager.parameters.STARVATION_PENALTY
        self._starvation_multiplier = (1.0 - penalty) ** self._starvation_steps

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
