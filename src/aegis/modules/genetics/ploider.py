class Ploider:
    """

    ### COMPUTATION ###

    This class compresses a diploid/polyploid genome into an array of haploid length.
    It does so depending on the dominance factor given by the user (parameter `DOMINANCE_FACTOR`).

    For homozygous sites, the compressed site has the same value as the input sites.

    For heterozygous sites, the compressed site will have the `DOMINANCE_FACTOR` value.
    The dominance factor will define the inheritance pattern.
    When the dominance factor is 0, the inheritance pattern is recessive; 0.5 is for true additive, 1 for dominant, 1+ for overdominant, and other values between 0 and 1 for partial dominant.
    The inheritance pattern will be applied to all sites (it is not gene-specific).
    """

    def __init__(self, REPRODUCTION_MODE, DOMINANCE_FACTOR):
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.DOMINANCE_FACTOR = DOMINANCE_FACTOR
        self.y = self.get_ploidy(REPRODUCTION_MODE)

    @staticmethod
    def get_ploidy(REPRODUCTION_MODE):
        return {
            "sexual": 2,
            "asexual": 1,
            "asexual_diploid": 2,
        }[REPRODUCTION_MODE]

    def diploid_to_haploid(self, loci):
        """Merge two arrays encoding two chromatids into one array.

        Arguments:
            loci: A bool numpy array with shape (population size, ploidy, genome length, BITS_PER_LOCUS)

        Returns:
            A bool numpy array with shape (population size, genome length, BITS_PER_LOCUS)
        """

        # compute homozygous (0, 1) or heterozygous (0.5)
        zygosity = loci.mean(1)
        mask = zygosity == 0.5

        # correct heterozygous with DOMINANCE_FACTOR
        if isinstance(self.DOMINANCE_FACTOR, list):
            zygosity[mask] = self.DOMINANCE_FACTOR[mask]
        else:
            zygosity[mask] = self.DOMINANCE_FACTOR

        return zygosity
