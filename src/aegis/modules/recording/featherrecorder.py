import logging
import pandas as pd
import numpy as np

from aegis.hermes import hermes
from aegis.modules.dataclasses.population import Population


class FeatherRecorder:
    def __init__(self, odir_genotypes, odir_phenotypes, odir_demography):
        self.odir_genotypes = odir_genotypes
        self.odir_phenotypes = odir_phenotypes
        self.odir_demography = odir_demography

    def write(self, population: Population):
        """Record demographic, genetic and phenotypic data from the current population."""
        if hermes.skip("SNAPSHOT_RATE") or len(population) == 0:
            return

        stage = hermes.get_stage()

        logging.debug(f"Snapshots recorded at stage {stage}")

        self.write_genotypes(stage=stage, population=population)
        self.write_phenotypes(stage=stage, population=population)
        self.write_demography(stage=stage, population=population)

    def write_genotypes(self, stage: int, population: Population):
        # TODO add more info to columns and rows
        """

        # OUTPUT SPECIFICATION
        format: feather
        content: complete snapshot of genomes of all individuals
        dtype: bool
        columns: int; site index
        rows: int; individual index
        """
        df_gen = pd.DataFrame(np.array(population.genomes.flatten()))
        df_gen.reset_index(drop=True, inplace=True)
        df_gen.columns = [str(c) for c in df_gen.columns]
        df_gen.to_feather(self.odir_genotypes / f"{stage}.feather")

    def write_phenotypes(self, stage: int, population: Population):
        # TODO add more info to columns and rows
        """

        # OUTPUT SPECIFICATION
        format: feather
        content: complete snapshot of phenotypes of all individuals
        dtype: float
        columns: int; phenotype index
        rows: int; individual index
        """
        # TODO bugged, wrong header
        df_phe = pd.DataFrame(np.array(population.phenotypes))
        df_phe.reset_index(drop=True, inplace=True)
        df_phe.columns = [str(c) for c in df_phe.columns]
        df_phe.to_feather(self.odir_phenotypes / f"{stage}.feather")

    def write_demography(self, stage: int, population: Population):
        # TODO add more info to columns and rows
        """

        # OUTPUT SPECIFICATION
        format: feather
        content: snapshot of previous life history of all individuals
        dtype: float
        columns: age, number of offspring, stage at which the individual was born, size, sex
        rows: int; individual index
        """
        dem_attrs = ["ages", "births", "birthdays", "sizes", "sexes"]
        demo = {attr: getattr(population, attr) for attr in dem_attrs}
        df_dem = pd.DataFrame(demo, columns=dem_attrs)
        df_dem.reset_index(drop=True, inplace=True)
        df_dem.to_feather(self.odir_demography / f"{stage}.feather")
