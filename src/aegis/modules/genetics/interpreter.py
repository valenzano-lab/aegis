"""Interpreter of genomic loci

Transforms bool array into an array of numbers.
These numbers can be loosely understood as gene activity.
"""
import numpy as np

from aegis.help import other

exp_base = 0.5  # Important for _exp
binary_exp_base = 0.98  # Important for _binary_exp

BITS_PER_LOCUS = None
BITS_PER_LOCUS = None
THRESHOLD = None
binary_weights = None
binary_switch_weights = None
linear_weights = None


def init(self, BITS_PER_LOCUS, DOMINANCE_FACTOR, THRESHOLD):
    self.DOMINANCE_FACTOR = DOMINANCE_FACTOR
    self.THRESHOLD = THRESHOLD

    # Parameters for the binary interpreter
    self.binary_weights = 2 ** np.arange(BITS_PER_LOCUS)[::-1]
    self.binary_weights = self.binary_weights / self.binary_weights.sum()

    # Parameters for the binary switch interpreter
    self.binary_switch_weights = 2 ** np.arange(BITS_PER_LOCUS)[::-1]
    self.binary_switch_weights[-1] = 0  # Switch bit does not add to locus value
    self.binary_switch_weights = self.binary_switch_weights / self.binary_switch_weights.sum()
    # e.g. when BITS_PER_LOCUS is 4, binary_switch_weights are [4/7, 2/7, 1/7, 0]

    # Parameters for the linear interpreter
    self.linear_weights = np.arange(BITS_PER_LOCUS)[::-1] + 1
    self.linear_weights = self.linear_weights / self.linear_weights.sum()


def _diploid_to_haploid(loci):
    """Merge two arrays encoding two chromatids into one array.

    Arguments:
        loci: A bool numpy array with shape (population size, ploidy, gstruc.length, BITS_PER_LOCUS)

    Returns:
        A bool numpy array with shape (population size, gstruc.length, BITS_PER_LOCUS)
    """

    # compute homozygous (0, 1) or heterozygous (0.5)
    zygosity = loci.mean(1)
    mask = zygosity == 0.5

    # correct heterozygous with DOMINANCE_FACTOR
    if isinstance(DOMINANCE_FACTOR, list):
        zygosity[mask] = DOMINANCE_FACTOR[mask]
    else:
        zygosity[mask] = DOMINANCE_FACTOR

    return zygosity


def _const1(loci):
    return np.ones((len(loci), 1))


def _single_bit(loci):
    return loci[:, :, 0]


def _threshold(loci):
    """Penna interpreter
    Cares only about the first bit of the locus.
    """
    return (~loci[:, :, 0]).cumsum(-1) < THRESHOLD


def _linear(loci):
    return np.matmul(loci, linear_weights)


def _binary(loci):
    """Interpret locus as a binary number and normalize.

    High resolution (can produce 2^bits_per_locus different numbers).
    Position-dependent.
    """

    return np.matmul(loci, binary_weights)


def _switch(loci):
    """Return 0 if all bits are 0; 1 if all bits are 1; 0 or 1 randomly otherwise.

    Low resolution (can produce 2 different numbers).
    Position-independent.
    """
    sums = loci.mean(2)
    rand_values = other.rng.random(loci.shape[:-1], dtype=np.float32) < 0.5
    return np.select([sums == 0, (sums > 0) & (sums < 1), sums == 1], [0, rand_values, 1])


def _binary_switch(loci):
    """Interpret first n-1 bits as a binary number if the last bit is 1.

    High resolution (can produce 2^(bits_per_locus-1) different numbers).
    Position-dependent.
    """
    where_on = loci[:, :, -1] == 1  # Loci which are turned on
    values = np.zeros(loci.shape[:-1], dtype=np.float32)  # Initialize output array with zeros
    values[where_on] = loci[where_on].dot(
        binary_switch_weights
    )  # If the locus is turned on, make the value in the output array be the binary value
    return values


def _uniform(loci):
    """Return normalized sum of bits.

    Medium resolution (can produce bits_per_locus+1 different numbers).
    Position-independent.
    """
    return loci.sum(-1) / loci.shape[-1]


def _exp(loci):
    """Return base^total_number_of_zeros.

    Medium resolution (can produce bits_per_locus+1 different numbers).
    Suitable for generating very small numbers.
    Position-independent.
    """
    return exp_base ** np.sum(1 - loci, axis=2)


def _binary_exp(loci):
    """Return base^binary_value_of_locus

    High resolution (can produce 2^bits_per_locus different numbers).
    Suitable for generating very small numbers.
    Position-dependent.
    """
    binary = _binary(loci)
    return binary_exp_base**binary


def do(loci, interpreter_kind):
    """Exposed method"""
    method = {
        "const1": _const1,
        "single_bit": _single_bit,
        "threshold": _threshold,
        "linear": _linear,
        "binary": _binary,
        "switch": _switch,
        "binary_switch": _binary_switch,
        "uniform": _uniform,
        "exp": _exp,
        "binary_exp": _binary_exp,
    }[interpreter_kind]
    if loci.shape[1] == 1:  # Do not calculate mean if genomes are haploid
        loci = loci[:, 0]
    else:
        loci = _diploid_to_haploid(loci)
    interpretome = method(loci)
    return interpretome
