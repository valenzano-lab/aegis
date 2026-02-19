"""Packed bit genome storage.

Stores genome data internally as np.uint8 packed arrays (8 bits per byte)
while preserving the same public interface as the original bool-based
implementation. This reduces genome memory by 8x.
"""

import numpy as np


class Genomes:
    def __init__(self, array, n_loci=None, bits_per_locus=None):
        """Initialize Genomes from a bool or uint8 array.

        Args:
            array: Either np.bool_ (will be packed) or np.uint8 (stored directly).
                   Bool arrays must be 4D with shape (n, ploidy, n_loci, bpl).
            n_loci: Number of loci (required for uint8 input, inferred for bool).
            bits_per_locus: Bits per locus (required for uint8 input, inferred for bool).
        """
        array = np.asarray(array)

        if array.dtype == np.uint8 and n_loci is not None and bits_per_locus is not None:
            # Uint8 input: store directly with provided metadata
            self._packed = array
            self._n_loci = n_loci
            self._bits_per_locus = bits_per_locus
            self._ploidy = array.shape[1] if array.ndim == 3 else 1
            self._n_packed_bytes = (n_loci * bits_per_locus + 7) // 8
        else:
            # Bool (or coercible) input: convert to bool, infer metadata, pack
            if array.dtype != np.bool_:
                array = array.astype(np.bool_)

            if array.ndim == 4:
                n, ploidy, nl, bpl = array.shape
                self._n_loci = nl
                self._bits_per_locus = bpl
                self._ploidy = ploidy
                self._n_packed_bytes = (nl * bpl + 7) // 8
                if n == 0:
                    self._packed = np.empty((0, ploidy, self._n_packed_bytes), dtype=np.uint8)
                else:
                    # Reshape to (n, ploidy, total_bits) then pack
                    flat = array.reshape(n, ploidy, -1)
                    self._packed = np.packbits(flat, axis=-1, bitorder='big')
            else:
                # Non-4D arrays (e.g. 2D, 3D from tests/legacy code):
                # Store as packed with minimal metadata
                total_bits = 1
                for d in array.shape[1:]:
                    total_bits *= d
                self._n_loci = total_bits
                self._bits_per_locus = 1
                self._ploidy = 1
                self._n_packed_bytes = (total_bits + 7) // 8
                self._original_shape = array.shape
                n = array.shape[0]
                if n == 0:
                    # Handle empty arrays
                    self._packed = np.empty((0, self._n_packed_bytes), dtype=np.uint8)
                else:
                    flat = array.reshape(n, -1)
                    self._packed = np.packbits(flat, axis=-1, bitorder='big')

    @property
    def array(self):
        """Backward-compatible property returning unpacked bool array.

        Returns the genome data in its original unpacked shape.
        """
        return self._unpack_original()

    @array.setter
    def array(self, value):
        """Backward-compatible setter that packs the assigned array."""
        new_g = Genomes(value)
        self._packed = new_g._packed
        self._n_loci = new_g._n_loci
        self._bits_per_locus = new_g._bits_per_locus
        self._ploidy = new_g._ploidy
        self._n_packed_bytes = new_g._n_packed_bytes
        if hasattr(new_g, '_original_shape'):
            self._original_shape = new_g._original_shape
        elif hasattr(self, '_original_shape'):
            del self._original_shape

    def _unpack_original(self):
        """Unpack to the original shape (4D or legacy shape)."""
        n = len(self._packed)
        if hasattr(self, '_original_shape'):
            # Non-4D legacy path
            if n == 0:
                return np.empty((0,) + self._original_shape[1:], dtype=np.bool_)
            total_bits = self._n_loci * self._bits_per_locus
            unpacked = np.unpackbits(self._packed, axis=-1, bitorder='big')
            # Trim padding bits
            unpacked = unpacked[:, :total_bits]
            target_shape = (n,) + self._original_shape[1:]
            return unpacked.reshape(target_shape).view(np.bool_)
        else:
            return self.unpack()

    def unpack(self):
        """Unpack to 4D bool array (n, ploidy, n_loci, bits_per_locus).

        Returns:
            np.bool_ array with shape (n, ploidy, n_loci, bits_per_locus).
        """
        n = len(self._packed)
        if n == 0:
            return np.empty((0, self._ploidy, self._n_loci, self._bits_per_locus), dtype=np.bool_)
        total_bits = self._n_loci * self._bits_per_locus
        unpacked = np.unpackbits(self._packed, axis=-1, bitorder='big')
        # Trim padding bits from last axis
        if self._packed.ndim == 3:
            unpacked = unpacked[:, :, :total_bits]
        else:
            unpacked = unpacked[:, :total_bits]
        return unpacked.reshape(n, self._ploidy, self._n_loci, self._bits_per_locus).view(np.bool_)

    def flatten(self):
        """Unpack and reshape to 2D (n_individuals, total_bits).

        Returns:
            np.bool_ array with shape (n, ploidy * n_loci * bits_per_locus).
        """
        if hasattr(self, '_original_shape'):
            return self._unpack_original().reshape(len(self), -1)
        return self.unpack().reshape(len(self), -1)

    def get_array(self):
        """Return a copy of the unpacked genome data.

        Returns:
            A copy of the bool array in its original shape.
        """
        return self._unpack_original().copy()

    def get_packed(self, individuals) -> np.ndarray:
        """Return packed uint8 rows for the given individuals.

        Args:
            individuals: Index array or boolean mask.

        Returns:
            np.ndarray of uint8 with shape (n, ploidy, n_packed_bytes).
            Returned array is a contiguous copy.
        """
        packed_rows = self._packed[individuals]
        if len(packed_rows) == 0:
            return np.empty(
                (0, self._ploidy, self._n_packed_bytes), dtype=np.uint8
            )
        return np.ascontiguousarray(packed_rows)

    @property
    def n_loci(self) -> int:
        """Number of loci in the genome."""
        return self._n_loci

    @property
    def bits_per_locus(self) -> int:
        """Bits per locus."""
        return self._bits_per_locus

    @property
    def n_packed_bytes(self) -> int:
        """Number of packed bytes per chromatid."""
        return self._n_packed_bytes



    def shape(self):
        """Return the logical unpacked shape.

        Returns:
            Tuple of (n_individuals, ploidy, n_loci, bits_per_locus) for 4D,
            or the original shape for legacy arrays.
        """
        if hasattr(self, '_original_shape'):
            return (len(self),) + self._original_shape[1:]
        return (len(self), self._ploidy, self._n_loci, self._bits_per_locus)

    def __len__(self):
        """Return the number of individuals."""
        return len(self._packed)

    def get(self, individuals):
        """Return unpacked bool rows for the given individuals.

        Args:
            individuals: Index array or boolean mask.

        Returns:
            np.ndarray of bool values in the original unpacked shape (minus first dim).
        """
        packed_rows = self._packed[individuals]
        n = len(packed_rows)
        if n == 0:
            if hasattr(self, '_original_shape'):
                return np.empty((0,) + self._original_shape[1:], dtype=np.bool_)
            return np.empty((0, self._ploidy, self._n_loci, self._bits_per_locus), dtype=np.bool_)
        # Unpack to bool for backward compatibility with downstream code
        total_bits = self._n_loci * self._bits_per_locus
        unpacked = np.unpackbits(packed_rows, axis=-1, bitorder='big')
        if packed_rows.ndim == 3:
            unpacked = unpacked[:, :, :total_bits]
        elif packed_rows.ndim == 2:
            unpacked = unpacked[:, :total_bits]
        if hasattr(self, '_original_shape'):
            return unpacked.reshape(n, *self._original_shape[1:]).view(np.bool_)
        return unpacked.reshape(n, self._ploidy, self._n_loci, self._bits_per_locus).view(np.bool_)

    def keep(self, individuals):
        """Filter the packed array in place.

        Args:
            individuals: Index array or boolean mask.
        """
        self._packed = self._packed[individuals]

    def add(self, genomes):
        """Concatenate another Genomes' packed data in place.

        Args:
            genomes: Another Genomes instance with compatible metadata.
        """
        self._packed = np.concatenate([self._packed, genomes._packed])

    def __getitem__(self, key):
        """Index the packed array and return a new Genomes.

        Args:
            key: Index, slice, or array for selection.

        Returns:
            New Genomes instance wrapping the selected packed data.
        """
        new_packed = self._packed[key]
        g = Genomes.__new__(Genomes)
        g._packed = new_packed
        g._n_loci = self._n_loci
        g._bits_per_locus = self._bits_per_locus
        g._ploidy = self._ploidy
        g._n_packed_bytes = self._n_packed_bytes
        if hasattr(self, '_original_shape'):
            g._original_shape = self._original_shape
        return g

    # TODO add logicalxor
