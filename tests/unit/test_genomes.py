"""Unit tests for the Genomes dataclass.

Covers: initialization and dtype coercion, __len__, flatten across
different dimensionalities, get (index/mask selection), __getitem__,
in-place add/keep, shape reporting, and copy semantics of get_array.
"""

import numpy as np
import pytest

from aegis_sim.dataclasses.genomes import Genomes


class TestGenomesInit:
    """Verify that Genomes coerces input arrays to bool dtype."""

    def test_stores_as_bool(self):
        """Integer input is cast to np.bool_ internally."""
        arr = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        g = Genomes(arr)
        assert g.array.dtype == np.bool_

    def test_preserves_values(self):
        """Bool input is stored unchanged."""
        arr = np.array([[True, False], [False, True]])
        g = Genomes(arr)
        np.testing.assert_array_equal(g.array, arr)


class TestGenomesLen:
    """Verify __len__ returns the number of individuals (first axis)."""

    def test_len_matches_first_dim(self):
        """len() equals the size of the first dimension."""
        g = Genomes(np.ones((5, 3), dtype=bool))
        assert len(g) == 5

    def test_len_empty(self):
        """An empty genome array has length 0."""
        g = Genomes(np.empty((0, 3), dtype=bool))
        assert len(g) == 0


class TestGenomesFlatten:
    """Verify flatten() collapses all dims after the first into one."""

    def test_flatten_2d(self):
        """2-D array stays the same shape after flatten."""
        g = Genomes(np.ones((4, 6), dtype=bool))
        flat = g.flatten()
        assert flat.shape == (4, 6)

    def test_flatten_3d(self):
        """3-D array (4, 2, 3) flattens to (4, 6)."""
        g = Genomes(np.ones((4, 2, 3), dtype=bool))
        flat = g.flatten()
        assert flat.shape == (4, 6)


class TestGenomesGet:
    """Verify get() returns a raw ndarray subset of the genome array."""

    def test_get_by_index(self):
        """Integer index array selects the correct rows."""
        arr = np.eye(3, dtype=bool)
        g = Genomes(arr)
        result = g.get(individuals=np.array([0, 2]))
        np.testing.assert_array_equal(result, arr[[0, 2]])

    def test_get_by_bool_mask(self):
        """Boolean mask selects matching individuals."""
        arr = np.eye(3, dtype=bool)
        g = Genomes(arr)
        mask = np.array([True, False, True])
        result = g.get(individuals=mask)
        assert result.shape[0] == 2

    def test_get_returns_array_not_genomes(self):
        """get() returns a plain ndarray, not a Genomes instance."""
        g = Genomes(np.ones((3, 2), dtype=bool))
        result = g.get(individuals=np.array([0]))
        assert isinstance(result, np.ndarray)


class TestGenomesGetitem:
    """Verify __getitem__ returns a new Genomes wrapping the sliced data."""

    def test_getitem_returns_genomes(self):
        """Slicing via [] produces a Genomes with the correct length."""
        g = Genomes(np.ones((3, 2), dtype=bool))
        sub = g[0:2]
        assert isinstance(sub, Genomes)
        assert len(sub) == 2


class TestGenomesAdd:
    """Verify add() concatenates another Genomes' array in place."""

    def test_add_concatenates(self):
        """After add(), length is the sum and values are preserved."""
        g1 = Genomes(np.ones((2, 3), dtype=bool))
        g2 = Genomes(np.zeros((3, 3), dtype=bool))
        g1.add(g2)
        assert len(g1) == 5
        assert g1.array[0, 0] == True
        assert g1.array[3, 0] == False


class TestGenomesKeep:
    """Verify keep() filters the genome array in place."""

    def test_keep_filters(self):
        """Integer index array retains only selected individuals."""
        g = Genomes(np.eye(4, dtype=bool))
        g.keep(individuals=np.array([1, 3]))
        assert len(g) == 2

    def test_keep_with_bool_mask(self):
        """Boolean mask retains only True-flagged individuals."""
        g = Genomes(np.eye(4, dtype=bool))
        g.keep(individuals=np.array([True, False, False, True]))
        assert len(g) == 2


class TestGenomesShape:
    """Verify shape() mirrors the underlying array shape."""

    def test_shape_matches_array(self):
        """4-D genome array reports its full shape."""
        arr = np.ones((5, 2, 10, 8), dtype=bool)
        g = Genomes(arr)
        assert g.shape() == (5, 2, 10, 8)


class TestGenomesGetArray:
    """Verify get_array() returns a defensive copy."""

    def test_returns_copy(self):
        """Mutating the returned copy does not affect the original."""
        arr = np.ones((3, 2), dtype=bool)
        g = Genomes(arr)
        copy = g.get_array()
        copy[0, 0] = False
        assert g.array[0, 0] == True  # original unchanged
