
import numpy as np

from aegis.panconfiguration import pan


class Interpreter:
    """Interpreter of genomic loci

    Transforms bool array into an array of numbers.
    These numbers can be loosely understood as gene activity.
    """

    exp_base = 0.5  # Important for _exp
    binary_exp_base = 0.98  # Important for _binary_exp

    def __init__(self, BITS_PER_LOCUS):

        # Parameters for the binary interpreter
        self.binary_weights = 2 ** np.arange(BITS_PER_LOCUS)[::-1]
        self.binary_max = self.binary_weights.sum()

        # Parameters for the binary switch interpreter
        self.binary_switch_weights = self.binary_weights.copy()
        self.binary_switch_weights[-1] = 0  # Switch bit does not add to locus value
        self.binary_switch_max = self.binary_switch_weights.sum()

    def __call__(self, loci, interpreter_kind):
        """Exposed method"""
        interpreter = getattr(self, f"_{interpreter_kind}")
        loci = self._diploid_to_haploid(loci)
        interpretome = interpreter(loci)
        return interpretome

    def _diploid_to_haploid(self, loci):
        """Merge two arrays encoding two chromatids into one array.

        The two chromatids contribute equally to bits, so that 1+1->1, 0+0->0 and 1+0->0.5 (as well as 0+1->0.5).

        Arguments:
            loci: A bool numpy array with shape (population size, ploidy, gstruc.length, BITS_PER_LOCUS)

        Returns:
            A bool numpy array with shape (population size, gstruc.length, BITS_PER_LOCUS)
        """
        return loci.mean(1)

    def _binary(self, loci):
        """Interpret locus as a binary number and normalize.

        High resolution (can produce 2^bits_per_locus different numbers).
        Position-dependent.
        """
        return loci.dot(self.binary_weights) / self.binary_max

    def _switch(self, loci):
        """Return 0 if all bits are 0; 1 if all bits are 1; 0 or 1 randomly otherwise.

        Low resolution (can produce 2 different numbers).
        Position-independent.
        """
        sums = loci.mean(2)
        rand_values = pan.rng.random(loci.shape[:-1]) < 0.5
        return np.select(
            [sums == 0, (sums > 0) & (sums < 1), sums == 1], [0, rand_values, 1]
        )

    def _binary_switch(self, loci):
        """Interpret first n-1 bits as a binary number if the last bit is 1.

        High resolution (can produce 2^(bits_per_locus-1) different numbers).
        Position-dependent.
        """
        where_on = loci[:, :, -1] == 1  # Loci which are turned on
        values = np.zeros(loci.shape[:-1], float)  # Initialize output array with zeros
        values[where_on] = (
            loci[where_on].dot(self.binary_switch_weights) / self.binary_switch_max
        )  # If the locus is turned on, make the value in the output array be the binary value
        return values

    def _uniform(self, loci):
        """Return normalized sum of bits.

        Medium resolution (can produce bits_per_locus+1 different numbers).
        Position-independent.
        """
        return loci.sum(-1) / loci.shape[-1]

    def _exp(self, loci):
        """Return base^total_number_of_zeros.

        Medium resolution (can produce bits_per_locus+1 different numbers).
        Suitable for generating very small numbers.
        Position-independent.
        """
        return self.exp_base ** np.sum(1 - loci, axis=2)

    def _binary_exp(self, loci):
        """Return base^binary_value_of_locus

        High resolution (can produce 2^bits_per_locus different numbers).
        Suitable for generating very small numbers.
        Position-dependent.
        """
        binary = self._binary(loci)
        return self.binary_exp_base ** binary
