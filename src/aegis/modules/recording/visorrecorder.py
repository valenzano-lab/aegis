import numpy as np

import pathlib

from aegis.hermes import hermes
from aegis.modules.dataclasses.population import Population

from .recorder import Recorder

class VisorRecorder(Recorder):

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "visor"
        self.init_odir()
        self.init_headers()

    def record(self, population):
        """Record data that is needed by visor."""
        # TODO rename VISOR_RATE into something more representative; potentially, restructure the recording rates
        if hermes.skip("VISOR_RATE") or len(population) == 0:
            return

        self.write_genotypes(population=population)
        self.write_phenotypes(population=population)

    def init_headers(self):
        with open(self.odir / "genotypes.csv", "ab") as f:
            length = hermes.architect.architecture.length
            ploidy = hermes.architect.architecture.ploid.y
            header0 = list(range(length)) * ploidy
            header1 = np.repeat(np.arange(ploidy), length)
            np.savetxt(f, [header0], delimiter=",", fmt="%i")
            np.savetxt(f, [header1], delimiter=",", fmt="%i")

        with open(self.odir / "phenotypes.csv", "ab") as f:
            age_limit = hermes.architect.architecture.AGE_LIMIT
            trait_list = list(hermes.traits.keys())
            n_traits = len(trait_list)
            header0 = np.repeat(trait_list, age_limit)
            header1 = list(np.arange(age_limit)) * n_traits
            np.savetxt(f, [header0], delimiter=",", fmt="%s")
            np.savetxt(f, [header1], delimiter=",", fmt="%i")

    def write_genotypes(self, population: Population):
        """
        genotypes.csv | Record allele frequency

        # OUTPUT SPECIFICATION
        filetype: csv
        domain: genotype
        short description:
        long description:
        content: allele frequencies across genome, across time
        dtype: float
        index: interval
        header: two levels -- bit index, ploidy
        column: site
        rows: interval
        path: /visor/genotypes.csv
        """
        with open(self.odir / "genotypes.csv", "ab") as f:
            array = population.genomes.flatten().mean(0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

    def write_phenotypes(self, population: Population):
        """
        phenotypes.csv | Record median phenotype

        # OUTPUT SPECIFICATION
        filetype: csv
        domain: phenotype
        short description:
        long description:
        content: median phenotypes, across time
        dtype: float
        index:
        header: two levels -- trait, age
        column:
        rows: interval
        path: /visor/genotypes.csv
        """
        with open(self.odir / "phenotypes.csv", "ab") as f:
            array = np.median(population.phenotypes, 0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")
