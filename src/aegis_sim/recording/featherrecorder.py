import logging
import pandas as pd
import numpy as np

import pathlib

from aegis_sim.dataclasses.population import Population
from .recorder import Recorder
from aegis_sim import variables

from aegis_sim import parameterization
from aegis_sim.parameterization import parametermanager
from aegis_sim.constants import GENETIC_TRAITS
from aegis_sim.utilities.funcs import steps_to_end, skip


class FeatherRecorder(Recorder):
    def __init__(self, odir: pathlib.Path):
        self.odir_genotypes = odir / "snapshots" / "genotypes"
        self.odir_phenotypes = odir / "snapshots" / "phenotypes"
        self.odir_demography = odir / "snapshots" / "demography"
        self.init_dir(self.odir_genotypes)
        self.init_dir(self.odir_phenotypes)
        self.init_dir(self.odir_demography)

    def write(self, population: Population):
        """Record demographic, genetic and phenotypic data from the current population."""

        # If not final snapshots to be taken, and about to skip or the population is extinct, do not write.
        final_snapshots = parametermanager.parameters.SNAPSHOT_FINAL_COUNT > steps_to_end()
        if not final_snapshots and (skip("SNAPSHOT_RATE") or len(population) == 0):
            return

        step = variables.steps

        logging.debug(f"Snapshots recorded at step {step}.")

        self.write_genotypes(step=step, population=population)
        self.write_phenotypes(step=step, population=population)
        self.write_demography(step=step, population=population)

    def write_genotypes(self, step: int, population: Population):
        """

        # OUTPUT SPECIFICATION
        path: /snapshots/genotypes/{step}.feather
        filetype: feather
        category: genotype
        description: A snapshot of complete binary genomes of all individuals at a certain simulation step.
        trait granularity: individual
        time granularity: snapshot
        frequency parameter: SNAPSHOT_RATE
        structure: A bool matrix; rows: individuals, columns: genome positions, values: bit states
        header: genome positions
        """
        if len(population) == 0:
            # Empty population can't be reshaped by genomes.flatten(); write empty feather
            df_gen = pd.DataFrame()
        else:
            df_gen = pd.DataFrame(np.array(population.genomes.flatten()))
            df_gen.columns = [str(c) for c in df_gen.columns]
        df_gen.reset_index(drop=True, inplace=True)
        df_gen.to_feather(self.odir_genotypes / f"{step}.feather")

    def write_phenotypes(self, step: int, population: Population):
        """

        # OUTPUT SPECIFICATION
        path: /snapshots/phenotypes/{step}.feather
        filetype: feather
        category: phenotype
        description: A snapshot of complete intrinsic phenotypes of all individuals at a certain simulation step.
        trait granularity: individual
        time granularity: snapshot
        frequency parameter: SNAPSHOT_RATE
        structure: A float matrix; rows: individuals, columns: {trait}_{age} for each evolvable trait, values: trait values
        header: {trait}_{age} e.g. surv_0, surv_1, ..., repr_0, repr_1, ...
        """
        df_phe = pd.DataFrame(population.phenotypes.get())
        df_phe.reset_index(drop=True, inplace=True)
        df_phe.columns = self._phenotype_columns()
        df_phe.to_feather(self.odir_phenotypes / f"{step}.feather")

    @staticmethod
    def _phenotype_columns():
        AGE_LIMIT = parametermanager.parameters.AGE_LIMIT
        cols = []
        for trait_name in GENETIC_TRAITS:
            trait = parameterization.traits[trait_name]
            if not trait.evolvable:
                continue
            if trait.agespecific:
                cols.extend(f"{trait_name}_{age}" for age in range(AGE_LIMIT))
            else:
                cols.append(trait_name)
        return cols

    def write_demography(self, step: int, population: Population):
        """

        # OUTPUT SPECIFICATION
        path: /snapshots/demography/{step}.feather
        filetype: feather
        category: demography
        description: A recording of life history metrics (age, number of births given, step at which born, current size, sex) of all individuals until a certain simulation step.
        trait granularity: individual
        time granularity: snapshot
        frequency parameter: SNAPSHOT_RATE
        structure: A matrix of ints and floats
        header: ['ages', 'births', 'birthdays', 'sizes', 'sexes']
        """
        dem_attrs = [
            "ages",
            "births",
            "birthdays",
            # "generations",
            "sizes",
            "sexes",
        ]
        demo = {attr: getattr(population, attr) for attr in dem_attrs}
        df_dem = pd.DataFrame(demo, columns=dem_attrs)
        df_dem.reset_index(drop=True, inplace=True)
        df_dem.to_feather(self.odir_demography / f"{step}.feather")
