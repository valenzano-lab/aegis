"""Contains functions for the computation of relevant population genetic metrics"""

import statistics
import itertools
import numpy as np


class PopgenStats:
    def __init__(self, REPR_MODE="sexual"):
        self.REPR_MODE = REPR_MODE

        self.pop_size_history = [10]

        self.allele_frequencies = None

    def analyze(self, genomes, g_muta_initial, g_muta_evolvable, gstruc, phenotypes):
        self.haploid_genomes = self._get_haploid_genomes(genomes)
        self.allele_frequencies = self.get_allele_frequencies()
        self.genotype_frequencies = self.get_genotype_frequencies(genomes)
        self.mean_h_per_bit = self.get_mean_h_per_bit()
        self.mean_h_per_bit_expected = self.get_mean_h_per_bit_expected()
        self.mu = self.get_mu(g_muta_initial, g_muta_evolvable, gstruc, phenotypes)
        self.segregating_sites = self.get_segregating_sites(genomes)

        self.theta_pi = self.get_theta_pi(genomes)  # TODO add other arguments
        self.theta_w = self.get_theta_w(genomes)  # TODO add other arguments
        self.reference_genome = self.get_reference_genome(genomes)
        self.sfs = self.get_sfs(genomes)  # TODO add other arguments
        self.theta_h = self.get_theta_h(genomes)  # TODO add other arguments

        print(self.reference_genome.shape)

        # self.theta =

        return (
            self.get_n(),
            self.get_ne(),
            # self.get_allele_frequencies(genomes),
        )

    @staticmethod
    def harmonic(i):
        """Returns the i-th harmonic number"""
        return np.sum([1 / x for x in np.arange(1, i + 1)])

    @staticmethod
    def harmonic_sq(i):
        """Returns the i-th harmonic square"""
        return np.sum([1 / (x ** 2) for x in np.arange(1, i + 1)])

    def _convert_diploid_to_staggered(self, genomes):
        """Returns genomes that have bits of first chromosome in odd positions, and bits of second chromosomes in even positions"""
        staggered = np.empty(
            shape=(genomes.shape[0], genomes.shape[2], genomes.shape[3] * 2),
            dtype=int,
        )
        staggered[:, :, ::2] = genomes[:, 0]
        staggered[:, :, 1::2] = genomes[:, 1]
        return

    def _get_haploid_genomes(self, genomes):
        """Returns list of chromosomes thus removing the second dimension"""
        return genomes.reshape(
            genomes.shape[0] * genomes.shape[1],
            genomes.shape[2],
            genomes.shape[3],
        )

    def get_n(self):
        """Returns the census population size N"""
        return self.pop_size_history[-1]

    def get_ne(self):
        """Returns the effective population size Ne"""
        return statistics.harmonic_mean(self.pop_size_history)

    def get_allele_frequencies(self):
        """Returns the frequency of the 1-allele at every position of the genome"""
        # Aligns the genomes of the population, disregarding chromosomes, and takes the mean
        return self.haploid_genomes.mean(0)

    def get_genotype_frequencies(self, genomes):
        """Output: [loc1_bit1_freq00, loc1_bit1_freq01, loc1_bit1_freq11, loc1_bit2_freq00, ...]"""
        if self.REPR_MODE == "asexual":
            return self.allele_frequencies

        # TODO: Fetch repr_mode from params/panconfiguration
        len_pop = genomes.shape[0]

        # Genotype = Sum of alleles at a position -> [0, 1, 2]
        # genotypes_raw = genomes.reshape(-1, 2).sum(1).reshape(len_pop, -1).transpose()
        genotypes_raw = genomes.sum(1).reshape(len_pop, -1).transpose()

        # Counts the frequencies of 0, 1 and 2 across the population
        genotype_freqs = (
            np.array([np.bincount(x, minlength=3) for x in genotypes_raw]).reshape(-1)
            / len_pop
        )

        return genotype_freqs

    def get_mean_h_per_bit(self):
        """Returns the mean heterozygosity per bit.
        Output: [Hloc1_bit1, Hloc1_bit2, ...] Entries: (bits_per_locus // 2) * nloci"""
        if self.REPR_MODE == "asexual":
            return None

        return self.genotype_frequencies[1::3]

    def get_mean_h_per_locus(self, genomes):
        """Returns the mean heterozygosity per locus.
        Output: [Hloc1, Hloc2, ...] Entries: nloci"""
        if self.REPR_MODE == "asexual":
            return None

        return self.mean_h_per_bit.reshape(-1, genomes.shape[2] >> 1).mean(1)

    def get_mean_h(self):
        """Returns the mean heterozygosity of a population.
        Output: H"""
        if self.REPR_MODE == "asexual":
            return None

        return self.mean_h_per_bit.mean()

    def get_mean_h_per_bit_expected(self):
        """Returns the expected mean heterozygosity per bit under Hardy-Weinberg-Equilibrium.
        Output: [Heloc1_bit1, Heloc1_bit2, ...] Entries: (bits_per_locus // 2) * nloci"""
        if self.REPR_MODE == "asexual":
            return None

        genotype_freqs_sqrd = self.genotype_frequencies ** 2
        sum_each_locus = genotype_freqs_sqrd.reshape(-1, 3).sum(1)
        return 1 - sum_each_locus

    def get_mean_h_expected(self):
        """Returns the expected mean heterozygosity per bit under Hardy-Weinberg-Equilibrium.
        Output: He"""
        if self.REPR_MODE == "asexual":
            return None

        return self.mean_h_per_bit_expected.mean()

    def get_mu(self, g_muta_initial, g_muta_evolvable, gstruc, phenotypes):
        """Return the mutation rate µ per gene per generation -> AEGIS-'Locus' interpreted as a gene"""
        if not g_muta_evolvable:
            return g_muta_initial

        return np.mean(phenotypes[:, gstruc["muta"].start])

    def get_theta(self, effective_pop_size, mutation_rate):
        """Returns the adjusted mutation rate theta = 4 * Ne * µ,
        where µ is the mutation rate per gene per generation and Ne is the effective population size"""
        ploidy_factor = 2 if self.REPR_MODE == "asexual" else 4
        theta = ploidy_factor * effective_pop_size * mutation_rate
        return theta

    def get_reference_genome(self, genomes):
        """Returns the reference genome based on which allele is most common at each position.
        Equal fractions -> 0"""
        # return np.round(genomes.reshape(genomes.shape[0], -1).mean(0)).astype("int32")
        return np.round(
            genomes.reshape(genomes.shape[0], genomes.shape[1], -1).mean(0)
        ).astype("int32")

    def get_segregating_sites(self, genomes):
        """Returns the number of segregating sites"""
        # Genomes are first aligned and summed at each site across the population
        # A site is segregating if its sum is not equal to either 0 or the population size N
        if self.REPR_MODE == "asexual":
            pre_segr_sites = self.haploid_genomes.reshape(
                self.haploid_genomes.shape[0], -1
            ).sum(0)
            segr_sites = (
                self.haploid_genomes.shape[1] * self.haploid_genomes.shape[2]
                - (pre_segr_sites == self.haploid_genomes.shape[0]).sum()
                - (pre_segr_sites == 0).sum()
            )
            return segr_sites

        genomes = self._convert_diploid_to_staggered(genomes)

        pre_segr_sites = (
            genomes.reshape(-1, 2).transpose().reshape(genomes.shape[0] << 1, -1).sum(0)
        )
        segr_sites = (
            ((genomes.shape[1] * genomes.shape[2]) >> 1)
            - (pre_segr_sites == genomes.shape[0] << 1).sum()
            + (pre_segr_sites == 0).sum()
        )
        return segr_sites

    def get_theta_w(self, genomes, sample_size=None, sample_provided=False):
        """Returns Watterson's estimator theta_w"""
        if self.REPR_MODE != "asexual":
            # The chromosomes get aligned
            # 3D (individuals, loci, bits): [1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4] ->
            #
            # 2D (everything, 2): [[1, 1, 1, 1, 3, 3, 3, 3],
            #                      [2, 2, 2, 2, 4, 4, 4, 4]] ->
            #
            # 3D (chromosomes, loci, bits // 2): [[1, 1, 1, 1],
            #    (individuals * 2, ...)           [2, 2, 2, 2],
            #                                     [3, 3, 3, 3],
            #                                     [4, 4, 4, 4]]
            genomes = (
                genomes.reshape(-1, 2)
                .transpose()
                .reshape(genomes.shape[0] << 1, genomes.shape[1], -1)
            )

        if sample_size is None:
            sample_size = genomes.shape[0]

        if sample_size < 2 or genomes.shape[0] < 2:
            return None

        if sample_provided:
            genomes_sample = genomes

        else:
            indices = np.random.choice(
                range(genomes.shape[0]), sample_size, replace=False
            )
            genomes_sample = genomes[indices, :, :]

        segr_sites = self.segregating_sites(genomes_sample)
        return segr_sites / self.harmonic(sample_size - 1)

    def get_theta_pi(self, genomes, sample_size=None, sample_provided=False):
        """Returns the estimator theta_pi (based on pairwise differences)"""
        if self.REPR_MODE != "asexual":
            # The chromosomes get aligned
            # 3D (individuals, loci, bits): [1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4] ->
            #
            # 2D (everything, 2): [[1, 1, 1, 1, 3, 3, 3, 3],
            #                      [2, 2, 2, 2, 4, 4, 4, 4]] ->
            #
            # 3D (chromosomes, loci, bits // 2): [[1, 1, 1, 1],
            #    (individuals * 2, ...)           [2, 2, 2, 2],
            #                                     [3, 3, 3, 3],
            #                                     [4, 4, 4, 4]]
            genomes = (
                genomes.reshape(-1, 2)
                .transpose()
                .reshape(genomes.shape[0] << 1, genomes.shape[1], -1)
            )

        if sample_size is None:
            sample_size = genomes.shape[0]

        if sample_size < 2 or genomes.shape[0] < 2:
            return None

        if sample_provided:
            genomes_sample = genomes

        else:
            indices = np.random.choice(
                range(genomes.shape[0]), sample_size, replace=False
            )
            genomes_sample = genomes[indices, :, :]

        combs = itertools.combinations(range(sample_size), 2)

        # Aligns chromosomes and count pairwise differences
        genomes_sample_flat = genomes_sample.reshape(sample_size, -1)
        diffs = np.array(
            [
                (genomes_sample_flat[i[0]] != genomes_sample_flat[i[1]]).sum()
                for i in combs
            ]
        )
        total_diffs = diffs.sum()
        ncomparisons = diffs.size

        return total_diffs / ncomparisons

    def tajimas_d(self, genomes, sample_size=None, sample_provided=False):
        """Returns Tajima's D"""
        if self.REPR_MODE != "asexual":
            # The chromosomes get aligned
            # 3D (individuals, loci, bits): [1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4] ->
            #
            # 2D (everything, 2): [[1, 1, 1, 1, 3, 3, 3, 3],
            #                      [2, 2, 2, 2, 4, 4, 4, 4]] ->
            #
            # 3D (chromosomes, loci, bits // 2): [[1, 1, 1, 1],
            #    (individuals * 2, ...)           [2, 2, 2, 2],
            #                                     [3, 3, 3, 3],
            #                                     [4, 4, 4, 4]]
            genomes = (
                genomes.reshape(-1, 2)
                .transpose()
                .reshape(genomes.shape[0] << 1, genomes.shape[1], -1)
            )

        if sample_size is None:
            sample_size = genomes.shape[0]

        if sample_size < 3 or genomes.shape[0] < 3:
            return None

        if sample_provided:
            genomes_sample = genomes

        else:
            indices = np.random.choice(
                range(genomes.shape[0]), sample_size, replace=False
            )
            genomes_sample = genomes[indices, :, :]

        pre_d = self.theta_pi(genomes_sample) - self.theta_w(genomes_sample)
        segr_sites = self.segregating_sites(genomes_sample)

        a_1 = self.harmonic(sample_size - 1)
        a_2 = self.harmonic_sq(sample_size - 1)
        b_1 = (sample_size + 1) / (3 * (sample_size - 1))
        b_2 = (2 * (sample_size ** 2 + sample_size + 3)) / (
            9 * sample_size * (sample_size - 1)
        )
        c_1 = b_1 - (1 / a_1)
        c_2 = b_2 - ((sample_size + 2) / (a_1 * sample_size)) + (a_2 / (a_1 ** 2))
        e_1 = c_1 / a_1
        e_2 = c_2 / ((a_1 ** 2) + a_2)
        d_stdev = ((e_1 * segr_sites) + (e_2 * segr_sites * (segr_sites - 1))) ** 0.5

        return pre_d / d_stdev

    def get_sfs(self, genomes, sample_size=None, sample_provided=False):
        """Returns the site frequency spectrum (allele frequency spectrum) of a sample"""
        if self.REPR_MODE != "asexual":
            # The chromosomes get aligned
            # 3D (individuals, loci, bits): [1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4] ->
            #
            # 2D (everything, 2): [[1, 1, 1, 1, 3, 3, 3, 3],
            #                      [2, 2, 2, 2, 4, 4, 4, 4]] ->
            #
            # 3D (chromosomes, loci, bits // 2): [[1, 1, 1, 1],
            #    (individuals * 2, ...)           [2, 2, 2, 2],
            #                                     [3, 3, 3, 3],
            #                                     [4, 4, 4, 4]]
            genomes = (
                genomes.reshape(-1, 2)
                .transpose()
                .reshape(genomes.shape[0] << 1, genomes.shape[1], -1)
            )

        if sample_size is None:
            sample_size = genomes.shape[0]

        if sample_size < 2 or genomes.shape[0] < 2:
            return None

        if sample_provided:
            genomes_sample = genomes

        else:
            indices = np.random.choice(
                range(genomes.shape[0]), sample_size, replace=False
            )
            genomes_sample = genomes[indices, :, :]

        ref = self.reference_genome
        pre_sfs = genomes_sample.reshape(genomes_sample.shape[0], -1).sum(0)
        pre_sfs[np.nonzero(ref)] -= sample_size
        pre_sfs = np.abs(pre_sfs)
        sfs = np.bincount(pre_sfs, minlength=sample_size + 1)[:-1]
        return sfs

    def get_theta_h(self, genomes, sample_size=None, sample_provided=False):
        """Returns Fay and Wu's estimator theta_h"""
        if self.REPR_MODE != "asexual":
            # The chromosomes get aligned
            # 3D (individuals, loci, bits): [1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4] ->
            #
            # 2D (everything, 2): [[1, 1, 1, 1, 3, 3, 3, 3],
            #                      [2, 2, 2, 2, 4, 4, 4, 4]] ->
            #
            # 3D (chromosomes, loci, bits // 2): [[1, 1, 1, 1],
            #    (individuals * 2, ...)           [2, 2, 2, 2],
            #                                     [3, 3, 3, 3],
            #                                     [4, 4, 4, 4]]
            genomes = (
                genomes.reshape(-1, 2)
                .transpose()
                .reshape(genomes.shape[0] << 1, genomes.shape[1], -1)
            )

        if sample_size is None:
            sample_size = genomes.shape[0]

        if sample_size < 2 or genomes.shape[0] < 2:
            return None

        if sample_provided:
            genomes_sample = genomes

        else:
            indices = np.random.choice(
                range(genomes.shape[0]), sample_size, replace=False
            )
            genomes_sample = genomes[indices, :, :]

        # sum from i=1 to i=n-1: ( (2 * S_i * i^2) / (n * (n-1)) )
        sfs = self.sfs(genomes_sample)
        t_h = (
            (2 * sfs * (np.arange(sample_size) ** 2))
            / (sample_size * (sample_size - 1))
        ).sum()
        return t_h

    def get_fayandwu_h(self, genomes, sample_size=None, sample_provided=False):
        """Returns Fay and Wu's H"""
        if self.REPR_MODE != "asexual":
            # The chromosomes get aligned
            # 3D (individuals, loci, bits): [1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4] ->
            #
            # 2D (everything, 2): [[1, 1, 1, 1, 3, 3, 3, 3],
            #                      [2, 2, 2, 2, 4, 4, 4, 4]] ->
            #
            # 3D (chromosomes, loci, bits // 2): [[1, 1, 1, 1],
            #    (individuals * 2, ...)           [2, 2, 2, 2],
            #                                     [3, 3, 3, 3],
            #                                     [4, 4, 4, 4]]
            genomes = (
                genomes.reshape(-1, 2)
                .transpose()
                .reshape(genomes.shape[0] << 1, genomes.shape[1], -1)
            )

        if sample_size is None:
            sample_size = genomes.shape[0]

        if sample_size < 2 or genomes.shape[0] < 2:
            return None

        if sample_provided:
            genomes_sample = genomes

        else:
            indices = np.random.choice(
                range(genomes.shape[0]), sample_size, replace=False
            )
            genomes_sample = genomes[indices, :, :]

        pre_h = self.theta_pi(genomes_sample) - self.theta_h(genomes_sample)
        h_stdev = 1  # TODO: Calculate actual variance of h
        return pre_h / h_stdev
