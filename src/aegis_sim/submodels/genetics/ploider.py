class Ploider:
    """ """

    def __init__(self, REPRODUCTION_MODE, DOMINANCE_FACTOR, PLOIDY):
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.DOMINANCE_FACTOR = DOMINANCE_FACTOR
        self.y = PLOIDY

        if REPRODUCTION_MODE == "sexual":
            assert PLOIDY == 2, f"If reproduction is sexual, ploidy can only be 2, not {PLOIDY}."

    def diploid_to_haploid(self, loci):
        """Merge two arrays encoding two chromatids into one array.

        Arguments:
            loci: A bool numpy array with shape (population size, ploidy, genome length, BITS_PER_LOCUS)

        Returns:
            A bool numpy array with shape (population size, genome length, BITS_PER_LOCUS)
        """

        assert len(loci.shape) == 4, len(loci.shape)  # e.g. (45, 2, 250, 8)
        assert loci.shape[1] in (1, 2), loci.shape[1]  # ploidy

        # TODO handle polyploidy too
        # compute homozygous (0, 1) or heterozygous (0.5)
        zygosity = loci.mean(1)
        mask = zygosity == 0.5

        # correct heterozygous with DOMINANCE_FACTOR
        if isinstance(self.DOMINANCE_FACTOR, list):
            zygosity[mask] = self.DOMINANCE_FACTOR[mask]
        else:
            zygosity[mask] = self.DOMINANCE_FACTOR

        assert len(zygosity.shape) == 3, len(zygosity.shape)

        return zygosity
