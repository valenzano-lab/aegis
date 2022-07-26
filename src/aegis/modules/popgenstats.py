"""Contains functions for the computation of relevant population genetic metrics"""

import statistics
import itertools
import logging
import numpy as np
from aegis.panconfiguration import pan


class PopgenStats:
    def __init__(self):
        self.pop_size_history = []

    def record_pop_size_history(self, genomes):
        """Records population sizes at last 1000 stages"""
        if len(self.pop_size_history) >= 1000:
            del self.pop_size_history[0]
        self.pop_size_history.append(len(genomes))

    def calc(self, input_genomes, mutation_rates):
        """Calculates all popgen metrics

        Set sample_size value to 0 or -1 to not perform any sampling.
        """

        # Infer ploidy from genomes
        self.ploidy = input_genomes.shape[1]

        # Data to process
        self.genomes = self.make_3D(input_genomes)
        self.gsample = self.get_genomes_sample()
        self.nsample = (
            0 if self.gsample is None else len(self.gsample)
        )  # TODO correct for ploidy?

        # Statistics on population
        self.n = self.get_n()
        self.ne = self.get_ne()
        self.allele_frequencies = self.get_allele_frequencies()
        self.genotype_frequencies = self.get_genotype_frequencies()
        self.mean_h_per_bit = self.get_mean_h_per_bit()
        self.mean_h_per_locus = self.get_mean_h_per_locus()
        self.mean_h = self.get_mean_h()
        self.mean_h_per_bit_expected = self.get_mean_h_per_bit_expected()
        self.mean_h_expected = self.get_mean_h_expected()
        self.mu = self.get_mu(mutation_rates)
        self.theta = self.get_theta()
        self.segregating_sites = self.get_segregating_sites(self.genomes, self.ploidy)
        self.reference_genome = self.get_reference_genome(self.genomes)

        # Statistics on sample
        if self.nsample:
            # TODO gsample is essentially haploid, check if that causes issues
            self.reference_genome_gsample = self.get_reference_genome(self.gsample)
            self.segregating_sites_gsample = self.get_segregating_sites(self.gsample, 1)
            self.theta_w = self.get_theta_w()
            self.theta_pi = self.get_theta_pi()
            self.tajimas_d = self.get_tajimas_d()
            self.sfs = self.get_sfs(
                self.reference_genome_gsample
            )  # Uses reference genome calculated from sample
            self.theta_h = self.get_theta_h()
            self.fayandwu_h = self.get_fayandwu_h()
        else:
            # TODO will cause Recorder to fail?
            (
                self.reference_genome_gsample,
                self.segregating_sites_gsample,
                self.theta_w,
                self.theta_pi,
                self.tajimas_d,
                self.sfs,
                self.theta_h,
                self.fayandwu_h,
            ) = [None] * 8

    def emit_simple(self):
        attrs = [
            "n",
            "ne",
            "mu",
            "segregating_sites",
            "segregating_sites_gsample",
            "theta",
            "theta_w",
            "theta_pi",
            "tajimas_d",
            "theta_h",
            "fayandwu_h",
        ]

        if self.ploidy == 2:
            attrs += ["mean_h", "mean_h_expected"]

        return {attr: getattr(self, attr) for attr in attrs}

    def emit_complex(self):
        attrs = [
            "allele_frequencies",
            "genotype_frequencies",
            "sfs",
            "reference_genome",
            "reference_genome_gsample",
        ]

        if self.ploidy == 2:
            attrs += ["mean_h_per_bit", "mean_h_per_locus", "mean_h_per_bit_expected"]

        return {attr: getattr(self, attr) for attr in attrs}

    ####################
    # HELPER FUNCTIONS #
    ####################

    @staticmethod
    def harmonic(i):
        """Returns the i-th harmonic number"""
        return (1 / np.arange(1, i + 1)).sum()

    @staticmethod
    def harmonic_sq(i):
        """Returns the i-th harmonic square"""
        return (1 / np.arange(1, i + 1) ** 2).sum()

    @staticmethod
    def make_3D(input_genomes):
        """Returns genomes array with merged chromosomes

        Methods of PopgenStats require the genomes to be in the form [individual, locus, bit] where, if individuals are diploid,
        the odd bits belong to a virtual first chromosome, while the even bits belong to a virtual second chromosome.
        If individuals are haploid, all bits belong to one virtual chromosome.

        This is contrast with the genomes arrays in the rest of AEGIS which is in form [individual, chromosome, locus, bit]
        so that all bits from one chromosome are in a separate array than those in the second chromosome.
        If individuals are haploid, the chromosome dimension contains only one element.
        """

        n_individuals, ploidy, n_loci, bits_per_locus = input_genomes.shape

        # TODO report when conversion cannot be executed

        if ploidy == 1:
            return input_genomes[:, 0]

        else:
            genomes = np.empty(
                shape=(n_individuals, n_loci, ploidy * bits_per_locus),
                dtype=np.bool8,
            )

            # Odd bits contain bits from chromosome 0
            genomes[:, :, 0::2] = input_genomes[:, 0]

            # Even bits contain bits from chromosome 1
            genomes[:, :, 1::2] = input_genomes[:, 1]

        return genomes

    @staticmethod
    def make_4D(genomes, ploidy):
        """Returns genomes array with chromosomes split along the second dimension"""

        n_individuals, n_loci, n_bits = genomes.shape
        bits_per_locus = n_bits // ploidy

        # TODO report when conversion cannot be executed

        unstaggered = np.empty(
            shape=(n_individuals, ploidy, n_loci, bits_per_locus),
            dtype=np.bool8,
        )

        if ploidy == 1:
            unstaggered[:, 0] = genomes
        if ploidy == 2:
            # Chromosome 0 contains odd bits from input genomes
            unstaggered[:, 0] = genomes[:, :, 0::2]
            # Chromosome 1 contains even bits from input genomes
            unstaggered[:, 1] = genomes[:, :, 1::2]

        return unstaggered

    def get_genomes_sample(self):
        """Returns a random sample of genomes"""
        if self.ploidy > 1:
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
                self.genomes.reshape(-1, 2)
                .transpose()
                .reshape(self.genomes.shape[0] * 2, self.genomes.shape[1], -1)
            )
        else:
            genomes = self.genomes

        # TODO check if change in ploidy has implications for popgen stats

        # Check if there are enough genomes to sample
        if len(genomes) < 2:  # NOTE tajimas_d requires minimum 2
            return None

        # Sample genomes
        if 0 < pan.POPGENSTATS_SAMPLE_SIZE_ <= genomes.shape[0]:
            indices = np.random.choice(
                range(genomes.shape[0]),
                pan.POPGENSTATS_SAMPLE_SIZE_,
                replace=False,
            )
            return genomes[indices]
        else:
            return genomes

    ####################################
    # OUTPUT: Statistics on population #
    ####################################

    def get_n(self):
        """Returns the census population size N"""
        return self.pop_size_history[-1]

    def get_ne(self):
        """Returns the effective population size Ne"""
        return statistics.harmonic_mean(self.pop_size_history)

    def get_allele_frequencies(self):
        """Returns the frequency of the 1-allele at every position of the genome"""
        # Aligns the genomes of the population, disregarding chromosomes, and takes the mean
        return self.genomes.reshape(self.genomes.shape[0], -1).mean(0)

    def get_genotype_frequencies(self):
        """Output: [loc1_bit1_freq00, loc1_bit1_freq01, loc1_bit1_freq11, loc1_bit2_freq00, ...]"""
        if self.ploidy == 1:
            return self.allele_frequencies

        len_pop = self.genomes.shape[0]

        # Genotype = Sum of alleles at a position -> [0, 1, 2]
        genotypes_raw = (
            self.genomes.reshape(-1, 2).sum(1).reshape(len_pop, -1).transpose()
        )

        # Counts the frequencies of 0, 1 and 2 across the population
        genotype_freqs = (
            np.array([np.bincount(x, minlength=3) for x in genotypes_raw]).reshape(-1)
            / len_pop
        )

        return genotype_freqs

    def get_mean_h_per_bit(self):
        """Returns the mean heterozygosity per bit.
        Output: [Hloc1_bit1, Hloc1_bit2, ...] Entries: (bits_per_locus // 2) * nloci"""
        if self.ploidy == 1:
            return None

        return self.genotype_frequencies[1::3]

    def get_mean_h_per_locus(self):
        """Returns the mean heterozygosity per locus.
        Output: [Hloc1, Hloc2, ...] Entries: nloci"""
        if self.ploidy == 1:
            return None

        h_per_bit = self.mean_h_per_bit
        return h_per_bit.reshape(-1, self.genomes.shape[2] // 2).mean(1)

    def get_mean_h(self):
        """Returns the mean heterozygosity of a population.
        Output: H"""
        if self.ploidy == 1:
            return None

        return self.mean_h_per_bit.mean()

    def get_mean_h_per_bit_expected(self):
        """Returns the expected mean heterozygosity per bit under Hardy-Weinberg-Equilibrium.
        Output: [Heloc1_bit1, Heloc1_bit2, ...] Entries: (bits_per_locus // 2) * nloci"""
        if self.ploidy == 1:
            return None

        genotype_freqs_sqrd = self.genotype_frequencies**2
        sum_each_locus = genotype_freqs_sqrd.reshape(-1, 3).sum(1)
        return 1 - sum_each_locus

    def get_mean_h_expected(self):
        """Returns the expected mean heterozygosity per bit under Hardy-Weinberg-Equilibrium.
        Output: He"""
        if self.ploidy == 1:
            return None

        return self.mean_h_per_bit_expected.mean()

    def get_mu(self, mutation_rates):
        """Return the mutation rate µ per gene per generation -> AEGIS-'Locus' interpreted as a gene"""
        return np.mean(mutation_rates)

    def get_theta(self):
        """Returns the adjusted mutation rate theta = 4 * Ne * µ,
        where µ is the mutation rate per gene per generation and Ne is the effective population size"""
        return (self.ploidy * 2) * self.ne * self.mu

    ##############################################
    # OUTPUT: Statistics on population or sample #
    ##############################################

    def get_reference_genome(self, genomes):
        """Returns the reference genome based on which allele is most common at each position.
        Equal fractions -> 0"""
        return np.round(genomes.reshape(genomes.shape[0], -1).mean(0)).astype("int32")

    def get_segregating_sites(self, genomes, ploidy):
        """Returns the number of segregating sites

        Genomes are first aligned and summed at each site across the population.
        A site is segregating if its sum is not equal to either 0 or the population size N.
        """

        if ploidy == 1:
            pre_segr_sites = genomes.reshape(genomes.shape[0], -1).sum(0)
            segr_sites = (
                genomes.shape[1] * genomes.shape[2]
                - (pre_segr_sites == genomes.shape[0]).sum()
                - (pre_segr_sites == 0).sum()
            )

            return segr_sites

        pre_segr_sites = (
            genomes.reshape(-1, 2).transpose().reshape(genomes.shape[0] * 2, -1).sum(0)
        )
        segr_sites = (
            ((genomes.shape[1] * genomes.shape[2]) // 2)
            - (pre_segr_sites == genomes.shape[0] * 2).sum()
            - (pre_segr_sites == 0).sum()
        )
        return segr_sites

    ################################
    # OUTPUT: Statistics on sample #
    ################################

    def get_theta_w(self):
        """Returns Watterson's estimator theta_w"""
        return self.segregating_sites_gsample / self.harmonic(self.nsample - 1)

    def get_theta_pi(self):
        """Returns the estimator theta_pi (based on pairwise differences)"""
        combs = itertools.combinations(range(self.nsample), 2)

        # Aligns chromosomes and count pairwise differences
        genomes_sample_flat = self.gsample.reshape(self.nsample, -1)
        diffs = np.array(
            [
                (genomes_sample_flat[i[0]] != genomes_sample_flat[i[1]]).sum()
                for i in combs
            ]
        )
        total_diffs = diffs.sum()
        ncomparisons = diffs.size

        return total_diffs / ncomparisons

    def get_tajimas_d(self):
        """Returns Tajima's D"""

        if self.nsample < 3:
            return None

        pre_d = self.theta_pi - self.theta_w
        segr_sites = self.segregating_sites_gsample

        if segr_sites == 0:
            logging.info(
                "Cannot compute Tajima's D because there are no segregating sites"
            )
            return

        a_1 = self.harmonic(self.nsample - 1)
        a_2 = self.harmonic_sq(self.nsample - 1)
        b_1 = (self.nsample + 1) / (3 * (self.nsample - 1))
        b_2 = (2 * (self.nsample**2 + self.nsample + 3)) / (
            9 * self.nsample * (self.nsample - 1)
        )
        c_1 = b_1 - (1 / a_1)
        c_2 = b_2 - ((self.nsample + 2) / (a_1 * self.nsample)) + (a_2 / (a_1**2))
        e_1 = c_1 / a_1
        e_2 = c_2 / ((a_1**2) + a_2)
        d_stdev = ((e_1 * segr_sites) + (e_2 * segr_sites * (segr_sites - 1))) ** 0.5

        return pre_d / d_stdev

    def get_sfs(self, reference_genome):
        """Returns the site frequency spectrum (allele frequency spectrum) of a sample"""
        pre_sfs = self.gsample.reshape(self.nsample, -1).sum(0)
        pre_sfs[np.nonzero(reference_genome)] -= self.nsample
        pre_sfs = np.abs(pre_sfs)
        sfs = np.bincount(pre_sfs, minlength=self.nsample + 1)[
            :-1
        ]  # TODO what if len(genomes) < sample_size
        return sfs

    def get_theta_h(self):
        """Returns Fay and Wu's estimator theta_h"""
        # sum from i=1 to i=n-1: ( (2 * S_i * i^2) / (n * (n-1)) )
        sfs = self.sfs
        t_h = (
            (2 * sfs * (np.arange(self.nsample) ** 2))
            / (self.nsample * (self.nsample - 1))
        ).sum()
        return t_h

    def get_fayandwu_h(self):
        """Returns Fay and Wu's H"""
        pre_h = self.theta_pi - self.theta_h
        h_stdev = 1  # TODO: Calculate actual variance of h
        return pre_h / h_stdev
