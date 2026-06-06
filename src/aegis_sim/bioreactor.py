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

        # Selection-coefficient experiment: forced allele introduction at a specific step.
        if variables.steps == parametermanager.parameters.ALLELE_INJECTION_STEP:
            self._inject_allele()

        # Selection-coefficient experiment: log allele frequency at the introduction locus.
        recordingmanager.selectionrecorder.write(self.population)

        # Mortality sources
        self.mortalities()
        resources.replenish()

        # Spatial lattice: migrate surviving individuals before reproduction.
        # Dead individuals' cells have been vacated by _kill via resync; this
        # step gives the survivors a chance to move into new cells. No-op
        # when LATTICE_MODE is False.
        if parametermanager.parameters.LATTICE_MODE and self.population.positions is not None:
            submodels.lattice.migrate(
                positions=self.population.positions,
                migration_rate=parametermanager.parameters.MIGRATION_RATE,
                migration_long_rate=parametermanager.parameters.MIGRATION_LONG_RATE,
            )

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
        recordingmanager.gvcfrecorder.write(self.population)
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

        # Lineage tracking — asexual only (sexual would need plumbing through pairing.py;
        # the warn-and-skip happens once at the start of the sim, not per step).
        offspring_lineage_id = None
        offspring_parent_lineage_id = None
        if parametermanager.parameters.LINEAGE_TRACING and self.population.lineage_id is not None:
            if parametermanager.parameters.REPRODUCTION_MODE == "asexual":
                # For asexual reproduction, len(offspring_genomes) == len(who), and
                # offspring[i] descends from parent at self.population[who[i]].
                if len(offspring_genomes) == len(who):
                    offspring_parent_lineage_id = self.population.lineage_id[who].astype(np.int64)
                    offspring_lineage_id = variables.next_lineage_ids(len(offspring_genomes))

        # Randomize order of newly laid egg attributes ..
        # .. because the order will affect their probability to be removed because of limited carrying capacity
        order = np.arange(len(offspring_sexes))
        variables.rng.shuffle(order)
        offspring_genomes = offspring_genomes[order]
        offspring_sexes = offspring_sexes[order]
        if offspring_ancestry is not None:
            offspring_ancestry = offspring_ancestry[order]
        if offspring_lineage_id is not None:
            offspring_lineage_id = offspring_lineage_id[order]
            offspring_parent_lineage_id = offspring_parent_lineage_id[order]
            recordingmanager.lineagerecorder.write_births(
                parent_lineage_ids=offspring_parent_lineage_id,
                child_lineage_ids=offspring_lineage_id,
                step=variables.steps,
            )

        # Make eggs
        eggs = Population.make_eggs(
            offspring_genomes=offspring_genomes,
            step=variables.steps,
            offspring_sexes=offspring_sexes,
            parental_generations=np.zeros(len(offspring_sexes)),  # TODO replace with working calculation
            offspring_ancestry=offspring_ancestry,
            offspring_lineage_id=offspring_lineage_id,
            offspring_parent_lineage_id=offspring_parent_lineage_id,
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

            # Lattice mode requires eggs to carry positions, assigned at reproduction
            # time. If they don't, fail loudly rather than silently corrupt the
            # lattice with (-1, -1) sentinel positions. (Offspring placement on
            # the lattice is the next commit.)
            if parametermanager.parameters.LATTICE_MODE and self.eggs.positions is None:
                raise NotImplementedError(
                    "LATTICE_MODE=True requires offspring placement on the lattice, "
                    "which is not yet wired into reproduction. Use LATTICE_MODE=False "
                    "for now, or wait for the lattice-reproduction commit."
                )

            self.eggs.phenotypes = submodels.architect.__call__(self.eggs.genomes)
            self.population += self.eggs
            self.eggs = None

            # Sync lattice occupancy after population growth so migration sees
            # all hatched individuals. No-op when LATTICE_MODE is False.
            if parametermanager.parameters.LATTICE_MODE and self.population.positions is not None:
                submodels.lattice.resync_occupancy_from_positions(self.population.positions)

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

        if self.population.lineage_id is not None:
            recordingmanager.lineagerecorder.write_deaths(
                lineage_ids=self.population.lineage_id[mask_kill],
                causeofdeath=causeofdeath,
                step=variables.steps,
            )

        # Retain survivors
        self.population *= ~mask_kill

        # Keep the lattice's occupancy grid consistent with population.positions
        # after the shrink. No-op when LATTICE_MODE is False.
        if parametermanager.parameters.LATTICE_MODE and self.population.positions is not None:
            submodels.lattice.resync_occupancy_from_positions(self.population.positions)

    def __len__(self):
        """Return the number of living individuals and saved eggs."""
        return len(self.population) + len(self.eggs) if self.eggs is not None else len(self.population)

    def _inject_allele(self):
        """Set ALLELE_INJECTION_ALLELE at the (TRAIT, AGE, BIT) locus on chromatid 0
        of a ALLELE_INJECTION_FRACTION-sized random subset of the living population.
        Recomputes phenotypes for the whole population so the new allele is expressed
        immediately."""
        from aegis_sim import parameterization

        n = len(self.population)
        if n == 0:
            logging.warning("ALLELE_INJECTION_STEP reached but population is empty; skipping.")
            return

        trait_name = parametermanager.parameters.ALLELE_INJECTION_TRAIT
        age = int(parametermanager.parameters.ALLELE_INJECTION_AGE)
        bit_in_locus = int(parametermanager.parameters.ALLELE_INJECTION_BIT)
        allele = bool(int(parametermanager.parameters.ALLELE_INJECTION_ALLELE))
        fraction = float(parametermanager.parameters.ALLELE_INJECTION_FRACTION)

        trait = parameterization.traits.get(trait_name)
        if trait is None or trait.length == 0:
            raise ValueError(
                f"ALLELE_INJECTION_TRAIT={trait_name!r} is not a valid evolvable trait (length=0)."
            )
        if trait.agespecific is True:
            if not (0 <= age < trait.length):
                raise ValueError(f"ALLELE_INJECTION_AGE={age} out of range [0, {trait.length}) for trait {trait_name}.")
            logical_locus = trait.start + age
        else:
            logical_locus = trait.start

        bits_per_locus = self.population.genomes.array.shape[-1]
        if not (0 <= bit_in_locus < bits_per_locus):
            raise ValueError(f"ALLELE_INJECTION_BIT={bit_in_locus} out of range [0, {bits_per_locus}).")

        physical_locus = int(submodels.architect.architecture.locus_permutation[logical_locus])

        n_carriers = max(1, int(round(n * fraction)))
        indices = variables.rng.choice(n, size=n_carriers, replace=False)
        self.population.genomes.array[indices, 0, physical_locus, bit_in_locus] = allele

        # Recompute phenotypes for the whole population so the new allele is expressed
        # by the affected individuals' current age slot (cheap; same call as init).
        self.population.phenotypes = submodels.architect(self.population.genomes)

        logging.info(
            "Mutation introduced at step %d: trait=%s age=%d bit=%d allele=%d in %d/%d individuals (chromatid 0, physical_locus=%d).",
            variables.steps, trait_name, age, bit_in_locus, int(allele), n_carriers, n, physical_locus,
        )
