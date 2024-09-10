import pandas as pd
import logging
import pathlib

from aegis_sim.hermes import hermes
from .recorder import Recorder


class PhenomapRecorder(Recorder):
    """

    Records once.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.init_odir()

    def write(self):
        """

        # OUTPUT SPECIFICATION
        path: /phenomap.csv
        filetype: csv
        keywords: genotype
        description: A static list of phenotypic effects of each genomic site. 
        structure: A table with four columns: effector site index, affected trait, affected age, effect magnitude. Each row represents an effect of a single site on a specific trait expressed at a specific age.
        """
        architecture = hermes.architect.architecture

        if hasattr(architecture, "phenomap"):
            phenolist = architecture.phenomap.phenolist
            pd.DataFrame(phenolist).to_csv(
                self.odir / "phenomap.csv", index=None, header=["bit", "trait", "age", "weight"]
            )
        else:
            logging.info("Phenomap is empty.")
