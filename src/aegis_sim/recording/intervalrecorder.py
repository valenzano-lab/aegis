import numpy as np
import pathlib

from aegis_sim.dataclasses.population import Population
from aegis_sim import parameterization

from aegis_sim.utilities.funcs import skip
from .recorder import Recorder
from aegis_sim import submodels


class IntervalRecorder(Recorder):

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "gui"
        self.init_odir()
        self.init_headers()

    def record(self, population):
        """Record data that is needed by gui."""
        # TODO rename INTERVAL_RATE into something more representative; potentially, restructure the recording rates
        if skip("INTERVAL_RATE") or len(population) == 0:
            return

        self.write_genotypes(population=population)
        self.write_phenotypes(population=population)

    def init_headers(self):
        with open(self.odir / "genotypes.csv", "ab") as f:
            length = submodels.architect.architecture.length
            ploidy = submodels.genetics.ploider.ploider.y
            header0 = list(range(length)) * ploidy
            header1 = np.repeat(np.arange(ploidy), length)
            np.savetxt(f, [header0], delimiter=",", fmt="%i")
            np.savetxt(f, [header1], delimiter=",", fmt="%i")

        with open(self.odir / "phenotypes.csv", "ab") as f:
            age_limit = submodels.architect.architecture.AGE_LIMIT
            trait_list = list(parameterization.traits.keys())
            n_traits = len(trait_list)
            header0 = np.repeat(trait_list, age_limit)
            header1 = list(np.arange(age_limit)) * n_traits
            np.savetxt(f, [header0], delimiter=",", fmt="%s")
            np.savetxt(f, [header1], delimiter=",", fmt="%i")

    def write_genotypes(self, population: Population):
        """
        genotypes.csv | Record allele frequency

        # OUTPUT SPECIFICATION
        path: /gui/genotypes.csv
        filetype: csv
        category: genotype
        description: A table of allele frequencies (frequency of 1's in the population for each site) across simulation intervals. Columns are sites, rows are simulation intervals (spanning INTERVAL_RATE steps).
        trait granularity: population mean
        time granularity: snapshot
        frequency parameter: INTERVAL_RATE
        structure: A float matrix.
        """
        with open(self.odir / "genotypes.csv", "ab") as f:
            array = population.genomes.flatten().mean(0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

    def write_phenotypes(self, population: Population):
        """
        phenotypes.csv | Record median phenotype

        # OUTPUT SPECIFICATION
        path: /gui/genotypes.csv
        filetype: csv
        category: phenotype
        description: A table of median intrinsic phenotypes (median phenotype rate for each trait at each age) across simulation intervals. Columns are traits, rows are simulation intervals (spanning INTERVAL_RATE steps).
        trait granularity: population median
        time granularity: snapshot
        frequency parameter: INTERVAL_RATE
        structure: A float matrix
        """
        with open(self.odir / "phenotypes.csv", "ab") as f:
            array = np.median(population.phenotypes.get(), 0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")
