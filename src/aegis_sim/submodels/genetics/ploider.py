import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _diploid_to_haploid_numba(c0, c1, dominance_factor):
    """Parallel numba kernel for diploid-to-haploid conversion.

    Operates on uint8 views of bool arrays to avoid numba's bool limitations.
    Returns float32 output: 1.0 for homozygous true, 0.0 for homozygous false,
    dominance_factor for heterozygous.
    """
    out = np.empty(c0.shape, dtype=np.float32)
    n0, n1, n2 = c0.shape
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                a = c0[i, j, k]
                b = c1[i, j, k]
                if a == b:
                    out[i, j, k] = np.float32(a)
                else:
                    out[i, j, k] = dominance_factor
    return out


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
            A float numpy array with shape (population size, genome length, BITS_PER_LOCUS)
        """

        assert len(loci.shape) == 4, len(loci.shape)  # e.g. (45, 2, 250, 8)
        assert loci.shape[1] == 2, loci.shape[1]  # ploidy

        arr = _diploid_to_haploid_numba(
            loci[:, 0].view(np.uint8),
            loci[:, 1].view(np.uint8),
            np.float32(self.DOMINANCE_FACTOR),
        )

        assert len(arr.shape) == 3, len(arr.shape)

        return arr


ploider = Ploider()
