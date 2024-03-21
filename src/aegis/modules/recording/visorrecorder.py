import numpy as np

from aegis.hermes import hermes


class VisorRecorder:
    def __init__(self, odir):
        self.odir = odir
        self.init_headers()

    def record(self, population):
        """Record data that is needed by visor."""
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
            array = np.arange(hermes.modules.architect.architecture.get_number_of_bits())
            np.savetxt(f, [array], delimiter=",", fmt="%i")

        with open(self.odir / "phenotypes.csv", "ab") as f:
            array = np.arange(hermes.modules.architect.architecture.get_number_of_phenotypic_values())
            np.savetxt(f, [array], delimiter=",", fmt="%i")
