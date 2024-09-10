"""Interpreter of genomic loci

Transforms bool array into an array of numbers.
These numbers can be loosely understood as gene activity.
"""

import numpy as np


class Interpreter:
    def __init__(self, BITS_PER_LOCUS, THRESHOLD):

        self.BITS_PER_LOCUS = BITS_PER_LOCUS
        self.THRESHOLD = THRESHOLD

        self.exp_base = 0.5  # Important for _exp
        self.binary_exp_base = 0.98  # Important for _binary_exp

        # Parameters for the binary interpreter
        binary_weights = 2 ** np.arange(self.BITS_PER_LOCUS)[::-1]
        self.binary_weights = binary_weights / binary_weights.sum()

        # Parameters for the binary switch interpreter
        binary_switch_weights = 2 ** np.arange(self.BITS_PER_LOCUS)[::-1]
        binary_switch_weights[-1] = 0  # Switch bit does not add to locus value
        self.binary_switch_weights = binary_switch_weights / binary_switch_weights.sum()
        # e.g. when BITS_PER_LOCUS is 4, binary_switch_weights are [4/7, 2/7, 1/7, 0]

        # Parameters for the linear interpreter
        linear_weights = np.arange(self.BITS_PER_LOCUS)[::-1] + 1
        self.linear_weights = linear_weights / linear_weights.sum()

    @staticmethod
    def _const1(loci):
        return np.ones((len(loci), 1))

    @staticmethod
    def _single_bit(loci):
        return loci[:, :, 0]

    def _threshold(self, loci):
        """Penna interpreter
        Cares only about the first bit of the locus.
        """
        return (~loci[:, :, 0]).cumsum(-1) < self.THRESHOLD

    def _linear(self, loci):
        return np.matmul(loci, self.linear_weights)

    def _binary(self, loci):
        """Interpret locus as a binary number and normalize.

        High resolution (can produce 2^bits_per_locus different numbers).
        Position-dependent.
        """

        return np.matmul(loci, self.binary_weights)

    @staticmethod
    def _switch(loci):
        """Return 0 if all bits are 0; 1 if all bits are 1; 0 or 1 randomly otherwise.

        Low resolution (can produce 2 different numbers).
        Position-independent.
        """
        sums = loci.mean(2)
        rand_values = np.random.random(loci.shape[:-1]) < 0.5
        return np.select([sums == 0, (sums > 0) & (sums < 1), sums == 1], [0, rand_values, 1])

    def _binary_switch(self, loci):
        """Interpret first n-1 bits as a binary number if the last bit is 1.

        High resolution (can produce 2^(bits_per_locus-1) different numbers).
        Position-dependent.
        """
        where_on = loci[:, :, -1] == 1  # Loci which are turned on
        values = np.zeros(loci.shape[:-1], dtype=np.float32)  # Initialize output array with zeros
        values[where_on] = loci[where_on].dot(
            self.binary_switch_weights
        )  # If the locus is turned on, make the value in the output array be the binary value
        return values

    def _uniform(loci):
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
        return self.binary_exp_base**binary

    def call(self, loci, interpreter_kind):
        """Exposed method"""
        method = {
            "const1": self._const1,
            "single_bit": self._single_bit,
            "threshold": self._threshold,
            "linear": self._linear,
            "binary": self._binary,
            "switch": self._switch,
            "binary_switch": self._binary_switch,
            "uniform": self._uniform,
            "exp": self._exp,
            "binary_exp": self._binary_exp,
        }[interpreter_kind]
        interpretome = method(loci)
        return interpretome
