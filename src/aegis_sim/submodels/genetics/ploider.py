import numpy as np


class Ploider:
    """ """

    def init(self, REPRODUCTION_MODE, DOMINANCE_FACTOR, PLOIDY):
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

        assert len(loci.shape) == 4, loci.shape  # e.g. (45, 2, 250, 8)
        assert loci.shape[1] == 2, loci.shape[1]  # ploidy

        # TODO handle polyploidy too
        # compute homozygous (0, 1) or heterozygous (0.5)

        # Three options: both 1, heterozygous, both 0

        # If at least one is 1 then 1; otherwise 0
        arr = np.logical_or(loci[:, 0], loci[:, 1]).astype(np.float64)

        # Heterozygous
        is_heterozygous = np.logical_xor(loci[:, 0], loci[:, 1])
        arr[is_heterozygous] = self.DOMINANCE_FACTOR

        assert len(arr.shape) == 3, len(arr.shape)

        return arr


ploider = Ploider()
