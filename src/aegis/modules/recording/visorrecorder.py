import numpy as np

from aegis.hermes import hermes


class VisorRecorder:
    def __init__(self, odir):
        self.odir = odir
        self.init_headers()

    def record(self, population):
        """Record data that is needed by visor."""
        # TODO rename VISOR_RATE into something more representative; potentially, restructure the recording rates
        if hermes.skip("VISOR_RATE") or len(population) == 0:
            return

        # genotypes.csv | Record allele frequency
        with open(self.odir / "genotypes.csv", "ab") as f:
            array = population.genomes.flatten().mean(0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

        # phenotypes.csv | Record median phenotype
        with open(self.odir / "phenotypes.csv", "ab") as f:
            array = np.median(population.phenotypes, 0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

    def init_headers(self):
        with open(self.odir / "genotypes.csv", "ab") as f:
            array = np.arange(hermes.architect.architecture.get_number_of_bits())
            np.savetxt(f, [array], delimiter=",", fmt="%i")

        with open(self.odir / "phenotypes.csv", "ab") as f:
            age_limit = hermes.architect.architecture.AGE_LIMIT
            trait_list = list(hermes.traits.keys())
            n_traits = len(trait_list)
            header0 = np.repeat(trait_list, age_limit)
            header1 = list(np.arange(age_limit)) * n_traits
            np.savetxt(f, [header0], delimiter=",", fmt="%s")
            np.savetxt(f, [header1], delimiter=",", fmt="%i")
