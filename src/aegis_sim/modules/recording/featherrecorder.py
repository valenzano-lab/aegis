import logging
import pandas as pd
import numpy as np

import pathlib

from aegis_sim.hermes import hermes
from aegis_sim.modules.dataclasses.population import Population
from .recorder import Recorder


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
        final_snapshots = hermes.parameters.SNAPSHOT_FINAL_COUNT > hermes.steps_to_end()
        if not final_snapshots and (hermes.skip("SNAPSHOT_RATE") or len(population) == 0):
            return

        step = hermes.get_step()

        logging.debug(f"Snapshots recorded at step {step}.")

        self.write_genotypes(step=step, population=population)
        self.write_phenotypes(step=step, population=population)
        self.write_demography(step=step, population=population)

    def write_genotypes(self, step: int, population: Population):
        """

        # OUTPUT SPECIFICATION
        path: /snapshots/genotypes/{step}.feather
        filetype: feather
        keywords: genotype
        description: A snapshot of complete binary genomes of all individuals at a certain simulation step.
        structure: A bool matrix
        """
        df_gen = pd.DataFrame(np.array(population.genomes.flatten()))
        df_gen.reset_index(drop=True, inplace=True)
        df_gen.columns = [str(c) for c in df_gen.columns]
        df_gen.to_feather(self.odir_genotypes / f"{step}.feather")

    def write_phenotypes(self, step: int, population: Population):
        # TODO add more info to columns and rows
        """

        # OUTPUT SPECIFICATION
        path: /snapshots/phenotypes/{step}.feather
        filetype: feather
        keywords: phenotype
        description: A snapshot of complete intrinsic phenotypes of all individuals at a certain simulation step.
        structure: A float matrix
        """
        # TODO bugged, wrong header
        df_phe = pd.DataFrame(np.array(population.phenotypes))
        df_phe.reset_index(drop=True, inplace=True)
        df_phe.columns = [str(c) for c in df_phe.columns]
        df_phe.to_feather(self.odir_phenotypes / f"{step}.feather")

    def write_demography(self, step: int, population: Population):
        """

        # OUTPUT SPECIFICATION
        path: /snapshots/demography/{step}.feather
        filetype: feather
        keywords: demography
        description: A recording of life history metrics (age, number of births given, step at which born, current size, sex) of all individuals until a certain simulation step.
        structure: A matrix of ints and floats
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
