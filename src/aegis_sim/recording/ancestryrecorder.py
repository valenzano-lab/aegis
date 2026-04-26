import pathlib
import numpy as np
import pandas as pd

from aegis_sim import variables, parameterization
from aegis_sim.parameterization import parametermanager
from aegis_sim.constants import GENETIC_TRAITS
from aegis_sim.utilities.funcs import skip
from .recorder import Recorder


class AncestryRecorder(Recorder):
    """Records mean introgression fraction per locus at each snapshot step.

    Output: snapshots/ancestry/{step}.csv
    Rows: one per step. Columns: {trait}_{age} for each evolvable trait.
    Values: mean fraction of introgressed alleles at that locus across all
    living individuals (averaged over ploidy and population).
    Only active when population.ancestry is not None (i.e. INTROGRESSION_SEEDS > 0).
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "snapshots" / "ancestry"
        self.init_dir(self.odir)

    def write(self, population):
        if population.ancestry is None:
            return
        if skip("SNAPSHOT_RATE") or len(population) == 0:
            return

        step = variables.steps

        # ancestry shape: (n, ploidy, n_loci_physical, bpl)
        # mean over ploidy and bpl dimensions → (n, n_loci_physical)
        locus_ancestry = population.ancestry.mean(axis=(1, 3))  # (n, n_loci_physical)
        # mean over individuals → (n_loci_physical,)
        mean_locus_physical = locus_ancestry.mean(axis=0)
        # Un-permute from physical to logical (trait×age) order
        from aegis_sim import submodels
        perm = submodels.architect.architecture.locus_permutation
        mean_locus = mean_locus_physical[perm]

        cols = self._locus_columns()
        df = pd.DataFrame([mean_locus], columns=cols)
        df.insert(0, "step", step)
        df.to_csv(self.odir / f"{step}.csv", index=False)

    @staticmethod
    def _locus_columns():
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
