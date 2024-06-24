import pandas as pd
import logging

from aegis.hermes import hermes


class PhenomapRecorder:
    """
    
    Records once.
    """
    def __init__(self, odir):
        self.odir = odir

    def write(self):
        """

        # OUTPUT SPECIFICATION
        format: csv
        content: genotype-phenotype mapping
        dtype: int
        index: none
        header: effector site index, affected trait, affected age, effect magnitude
        column: a specification
        rows: one effect of one site
        """
        architecture = hermes.architect.architecture

        if hasattr(architecture, "phenomap"):
            phenolist = architecture.phenomap.phenolist
            pd.DataFrame(phenolist).to_csv(
                self.odir / "phenomap.csv", index=None, header=["bit", "trait", "age", "weight"]
            )
        else:
            logging.info("Phenomap is empty")
