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
        description: Simple population metrics including population size, effective population size, mu, segregating sites, segregating sites using a genomic sample, theta, theta_w, theta_pi, tajimas_d, theta_h, and fayandwu_h.
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/allele_frequencies.csv
        filetype: csv
        keywords: population genetics
        description: 1-allele population-frequencies of every genomic site.
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/genotype_frequencies.csv
        filetype: csv
        keywords: population genetics
        description: Genotype frequencies at site resolution (e.g. for a diploid genome, number of 00, 01 and 11 for each site).
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/sfs.csv
        filetype: csv
        keywords: population genetics
        description: A site frequency spectrum.
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/mean_h_per_bit_expected.csv
        filetype: csv
        keywords: population genetics
        description: Heterozygosity per bit.
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/mean_h_per_bit.csv
        filetype: csv
        keywords: population genetics
        description: Expected mean heterozygosity per bit under Hardy-Weinberg-Equilibrium.
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/mean_h_per_locus.csv
        filetype: csv
        keywords: population genetics
        description: Mean bit heterozygosity per locus.
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/reference_genome.csv
        filetype: csv
        keywords: population genetics
        description: Reference genome based on which allele is most common at each position.
        structure: A float matrix.

        # OUTPUT SPECIFICATION
        path: /popgen/reference_genome_gsample.csv
        filetype: csv
        keywords: population genetics
        description: Reference genome based on which allele is most common at each position in a sample of genomes.
        structure: A float matrix.
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

        # TODO when writing some metrics (e.g. reference genome, reference genome gsample) use the appropriate dtype (bool in that case)

        # Record complex statistics
        complex_statistics = hermes.modules.popgenstats.emit_complex()
        for key, array in complex_statistics.items():
            with open(self.odir / f"{key}.csv", "ab") as f:
                np.savetxt(f, [array], delimiter=",", fmt="%1.3e")
