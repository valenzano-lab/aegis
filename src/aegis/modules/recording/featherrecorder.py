import logging
import pandas as pd
import numpy as np

from aegis.hermes import hermes


class FeatherRecorder:
    def __init__(self, odir_genotypes, odir_phenotypes, odir_demography):
        self.odir_genotypes = odir_genotypes
        self.odir_phenotypes = odir_phenotypes
        self.odir_demography = odir_demography

    def write(self, population):
        """Record demographic, genetic and phenotypic data from the current population."""
        if hermes.skip("SNAPSHOT_RATE") or len(population) == 0:
            return

        stage = hermes.get_stage()

        logging.debug(f"Snapshots recorded at stage {stage}")

        # genotypes
        df_gen = pd.DataFrame(np.array(population.genomes.flatten()))
        df_gen.reset_index(drop=True, inplace=True)
        df_gen.columns = [str(c) for c in df_gen.columns]
        df_gen.to_feather(self.odir_genotypes / f"{stage}.feather")

        # phenotypes
        df_phe = pd.DataFrame(np.array(population.phenotypes))
        df_phe.reset_index(drop=True, inplace=True)
        df_phe.columns = [str(c) for c in df_phe.columns]
        df_phe.to_feather(self.odir_phenotypes / f"{stage}.feather")

        # demography
        dem_attrs = ["ages", "births", "birthdays", "sizes"]
        demo = {attr: getattr(population, attr) for attr in dem_attrs}
        df_dem = pd.DataFrame(demo, columns=dem_attrs)
        df_dem.reset_index(drop=True, inplace=True)
        df_dem.to_feather(self.odir_demography / f"{stage}.feather")
