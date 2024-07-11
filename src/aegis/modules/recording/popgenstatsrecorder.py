import numpy as np

from aegis.hermes import hermes


from .recorder import Recorder


class PopgenStatsRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir / "popgen"
        self.init_odir()

    def write(self, genomes, mutation_rates):
        """
        Record population size in popgenstats, and record popgen statistics

        # OUTPUT SPECIFICATION
        path: /popgen/simple.csv
        filetype: csv
        keywords: population genetics
        structure: A matrix containing the population size, effective population size, mu, segregating sites, segregating sites using a genomic sample, theta, theta_w, theta_pi, tajimas_d, theta_h, and fayandwu_h.

        # OUTPUT SPECIFICATION
        path: /popgen/allele_frequencies.csv
        filetype: csv
        keywords: population genetics
        structure:

        # OUTPUT SPECIFICATION
        path: /popgen/genotype_frequencies.csv
        filetype: csv
        keywords: population genetics
        structure:

        # OUTPUT SPECIFICATION
        path: /popgen/sfs.csv
        filetype: csv
        keywords: population genetics
        structure:

        # OUTPUT SPECIFICATION
        path: /popgen/reference_genome.csv
        filetype: csv
        keywords: population genetics
        structure:

        # OUTPUT SPECIFICATION
        path: /popgen/reference_genome_gsample.csv
        filetype: csv
        keywords: population genetics
        structure:
        """
        hermes.modules.popgenstats.record_pop_size_history(genomes.array)

        if hermes.skip("POPGENSTATS_RATE") or len(genomes) == 0:
            return

        hermes.modules.popgenstats.calc(genomes.array, mutation_rates)

        # Record simple statistics
        array = list(hermes.modules.popgenstats.emit_simple().values())
        if None in array:
            return

        with open(self.odir / "simple.csv", "ab") as f:
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

        # Record complex statistics
        complex_statistics = hermes.modules.popgenstats.emit_complex()
        for key, array in complex_statistics.items():
            with open(self.odir / f"{key}.csv", "ab") as f:
                np.savetxt(f, [array], delimiter=",", fmt="%1.3e")
