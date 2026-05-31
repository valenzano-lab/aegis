import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _diploid_to_haploid_numba(c0, c1, dominance_per_locus):
    """Parallel numba kernel for diploid-to-haploid conversion.

    Operates on uint8 views of bool arrays to avoid numba's bool limitations.
    Returns float32 output: 1.0 for homozygous true, 0.0 for homozygous false,
    dominance_per_locus[j] for heterozygous (j = locus index along axis 1).
    """
    out = np.empty(c0.shape, dtype=np.float32)
    n0, n1, n2 = c0.shape
    for i in prange(n0):
        for j in range(n1):
            h = dominance_per_locus[j]
            for k in range(n2):
                a = c0[i, j, k]
                b = c1[i, j, k]
                if a == b:
                    out[i, j, k] = np.float32(a)
                else:
                    out[i, j, k] = h
    return out


class Ploider:
    """ """

    def init(self, REPRODUCTION_MODE, DOMINANCE_FACTOR, PLOIDY):
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        # Kept for backward-compatible config parsing; no longer used by the
        # collapse kernel. See G_<trait>_dominance for per-trait control.
        self.DOMINANCE_FACTOR = DOMINANCE_FACTOR
        self.y = PLOIDY

        if REPRODUCTION_MODE == "sexual":
            assert PLOIDY == 2, f"If reproduction is sexual, ploidy can only be 2, not {PLOIDY}."

    def diploid_to_haploid(self, loci, dominance_per_locus=None):
        """Merge two arrays encoding two chromatids into one array.

        Arguments:
            loci: A bool numpy array with shape (population size, ploidy, genome length, BITS_PER_LOCUS)
            dominance_per_locus: float32 array of shape (genome length,) giving h per locus.
                If None, a uniform array of 0.5 (codominant) is used.

        Returns:
            A float numpy array with shape (population size, genome length, BITS_PER_LOCUS)
        """

        assert len(loci.shape) == 4, len(loci.shape)  # e.g. (45, 2, 250, 8)
        assert loci.shape[1] == 2, loci.shape[1]  # ploidy

        n_loci = loci.shape[2]
        if dominance_per_locus is None:
            dominance_per_locus = np.full(n_loci, 0.5, dtype=np.float32)
        else:
            assert dominance_per_locus.shape == (n_loci,), (
                f"dominance_per_locus shape {dominance_per_locus.shape} != ({n_loci},)"
            )
            if dominance_per_locus.dtype != np.float32:
                dominance_per_locus = dominance_per_locus.astype(np.float32)

        arr = _diploid_to_haploid_numba(
            loci[:, 0].view(np.uint8),
            loci[:, 1].view(np.uint8),
            dominance_per_locus,
        )

        assert len(arr.shape) == 3, len(arr.shape)

        return arr


ploider = Ploider()
