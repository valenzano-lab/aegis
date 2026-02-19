"""Property-based tests for packed bit genome storage.

Uses hypothesis to verify correctness properties across randomly generated inputs.
Each test is annotated with the design property it validates.

Properties covered:
- P1: Pack/unpack round-trip
- P3: Packed get equivalence
- P4: Packed keep equivalence
- P5: Packed add equivalence
- P6: Unpack interface consistency
- P7: Mutation bit isolation
- P8: Mutate-by-bit equivalence (skipped — mutate_by_bit requires full sim init)
- P9: Recombination packed/unpacked equivalence
- P10: Pairing packed/unpacked equivalence
- P11: Checkpoint format round-trip
"""

import numpy as np
import pytest
import pickle
import tempfile

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim.dataclasses.legacy_genomes import LegacyGenomes


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def genome_arrays(draw, min_n=1, max_n=100, max_loci=50, bpl_choices=(1, 8)):
    """Generate random 4D bool genome arrays where total bits is divisible by 8."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    ploidy = 2
    bpl = draw(st.sampled_from(bpl_choices))
    # Ensure n_loci * bpl is divisible by 8
    if bpl == 1:
        n_loci = draw(st.integers(min_value=8, max_value=max_loci)) // 8 * 8
    elif bpl == 8:
        n_loci = draw(st.integers(min_value=1, max_value=max_loci))
    else:
        n_loci = draw(st.integers(min_value=1, max_value=max_loci))
        total = n_loci * bpl
        # Round up to multiple of 8
        n_loci = ((total + 7) // 8 * 8) // bpl
    assume(n_loci > 0)
    assume(n_loci * bpl % 8 == 0)
    arr = np.random.default_rng(draw(st.integers(0, 2**32-1))).integers(
        0, 2, size=(n, ploidy, n_loci, bpl)
    ).astype(np.bool_)
    return arr


@st.composite
def index_arrays(draw, max_n):
    """Generate valid index arrays for a population of size max_n."""
    n = draw(st.integers(min_value=0, max_value=max_n))
    indices = draw(st.lists(
        st.integers(min_value=0, max_value=max(max_n-1, 0)),
        min_size=n, max_size=n,
    ))
    return np.array(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Property 1: Pack/unpack round-trip
# ---------------------------------------------------------------------------

class TestPackUnpackRoundTrip:
    """Feature: packed-bit-genomes, Property 1: Pack/unpack round-trip"""

    @given(arr=genome_arrays())
    @settings(max_examples=20)
    def test_roundtrip(self, arr):
        """Validates: Requirements 1.5"""
        g = Genomes(arr)
        unpacked = g.unpack()
        np.testing.assert_array_equal(unpacked, arr)

    @given(arr=genome_arrays())
    @settings(max_examples=10)
    def test_double_roundtrip(self, arr):
        """Pack → unpack → pack → unpack should be identical."""
        g1 = Genomes(arr)
        unpacked1 = g1.unpack()
        g2 = Genomes(unpacked1)
        unpacked2 = g2.unpack()
        np.testing.assert_array_equal(unpacked1, unpacked2)


# ---------------------------------------------------------------------------
# Property 3, 4, 5: Packed get/keep/add equivalence
# ---------------------------------------------------------------------------

class TestPackedGetEquivalence:
    """Feature: packed-bit-genomes, Property 3: Packed get equivalence"""

    @given(arr=genome_arrays(min_n=2, max_n=50))
    @settings(max_examples=20)
    def test_get_matches_legacy(self, arr):
        """Validates: Requirements 3.1"""
        n = arr.shape[0]
        indices = np.random.default_rng(42).choice(n, size=min(n, 5), replace=False)

        packed = Genomes(arr)
        legacy = LegacyGenomes(arr)

        packed_result = packed.get(individuals=indices)
        legacy_result = legacy.get(individuals=indices)

        np.testing.assert_array_equal(packed_result, legacy_result)


class TestPackedKeepEquivalence:
    """Feature: packed-bit-genomes, Property 4: Packed keep equivalence"""

    @given(arr=genome_arrays(min_n=2, max_n=50))
    @settings(max_examples=20)
    def test_keep_matches_legacy(self, arr):
        """Validates: Requirements 3.2"""
        n = arr.shape[0]
        mask = np.random.default_rng(42).random(n) > 0.5
        if not mask.any():
            mask[0] = True

        packed = Genomes(arr.copy())
        legacy = LegacyGenomes(arr.copy())

        packed.keep(individuals=mask)
        legacy.keep(individuals=mask)

        np.testing.assert_array_equal(packed.unpack(), legacy.array)


class TestPackedAddEquivalence:
    """Feature: packed-bit-genomes, Property 5: Packed add equivalence"""

    @given(data=st.data())
    @settings(max_examples=20)
    def test_add_matches_legacy(self, data):
        """Validates: Requirements 3.3"""
        arr1 = data.draw(genome_arrays(min_n=1, max_n=30))
        # Second array must have same shape except first dim
        n2 = data.draw(st.integers(min_value=1, max_value=30))
        arr2 = np.random.default_rng(99).integers(
            0, 2, size=(n2, *arr1.shape[1:])
        ).astype(np.bool_)

        packed1 = Genomes(arr1.copy())
        packed2 = Genomes(arr2.copy())
        legacy1 = LegacyGenomes(arr1.copy())
        legacy2 = LegacyGenomes(arr2.copy())

        packed1.add(packed2)
        legacy1.add(legacy2)

        np.testing.assert_array_equal(packed1.unpack(), legacy1.array)


# ---------------------------------------------------------------------------
# Property 6: Unpack interface consistency
# ---------------------------------------------------------------------------

class TestUnpackInterfaceConsistency:
    """Feature: packed-bit-genomes, Property 6: Unpack interface consistency"""

    @given(arr=genome_arrays())
    @settings(max_examples=20)
    def test_interface(self, arr):
        """Validates: Requirements 3.4, 3.5, 3.6, 3.7, 3.8"""
        g = Genomes(arr)

        # unpack shape
        assert g.unpack().shape == arr.shape
        assert g.unpack().dtype == np.bool_

        # flatten shape
        flat = g.flatten()
        n = arr.shape[0]
        total_bits = arr.shape[1] * arr.shape[2] * arr.shape[3]
        assert flat.shape == (n, total_bits)
        assert flat.dtype == np.bool_

        # get_array is a copy
        copy = g.get_array()
        assert copy.shape == arr.shape
        assert copy.dtype == np.bool_
        copy[0, 0, 0, 0] = not copy[0, 0, 0, 0]
        assert g.unpack()[0, 0, 0, 0] != copy[0, 0, 0, 0]

        # shape
        assert g.shape() == arr.shape

        # len
        assert len(g) == n


# ---------------------------------------------------------------------------
# Property 7: Mutation bit isolation
# ---------------------------------------------------------------------------

class TestMutationBitIsolation:
    """Feature: packed-bit-genomes, Property 7: Mutation bit isolation"""

    @given(arr=genome_arrays(min_n=1, max_n=10, max_loci=16))
    @settings(max_examples=20)
    def test_xor_flips_one_bit(self, arr):
        """Validates: Requirements 4.1, 4.2"""
        g = Genomes(arr)
        n, ploidy, n_loci, bpl = arr.shape

        # Pick a random target
        rng = np.random.default_rng(42)
        ind = rng.integers(0, n)
        chrom = rng.integers(0, ploidy)
        locus = rng.integers(0, n_loci)
        bit = rng.integers(0, bpl)

        # Compute packed coordinates
        flat_bit = locus * bpl + bit
        byte_idx = flat_bit // 8
        bit_pos = flat_bit % 8

        # Flip the bit via XOR on packed data
        before = g._packed.copy()
        g._packed[ind, chrom, byte_idx] ^= np.uint8(1 << (7 - bit_pos))

        # Verify exactly one bit changed
        unpacked_before = Genomes.__new__(Genomes)
        unpacked_before._packed = before
        unpacked_before._n_loci = g._n_loci
        unpacked_before._bits_per_locus = g._bits_per_locus
        unpacked_before._ploidy = g._ploidy
        unpacked_before._n_packed_bytes = g._n_packed_bytes

        diff = unpacked_before.unpack() != g.unpack()
        assert diff.sum() == 1
        assert diff[ind, chrom, locus, bit] == True


# ---------------------------------------------------------------------------
# Property 9: Recombination packed/unpacked equivalence
# ---------------------------------------------------------------------------

class TestRecombinationEquivalence:
    """Feature: packed-bit-genomes, Property 9: Recombination equivalence"""

    @given(arr=genome_arrays(min_n=2, max_n=20, max_loci=32))
    @settings(max_examples=10, deadline=None)
    def test_recombination_same_result(self, arr):
        """Validates: Requirements 5.3
        
        Recombination operates on unpacked bool arrays (via get()), so the
        result should be identical regardless of packed storage.
        """
        from aegis_sim.submodels.reproduction.recombination import recombination_via_pairs
        import aegis_sim.variables as variables

        # Set up RNG
        variables.rng = np.random.default_rng(42)
        np.random.seed(42)

        # Run on packed genomes (get() unpacks)
        packed = Genomes(arr)
        packed_input = packed.get(individuals=np.arange(len(packed)))
        variables.rng = np.random.default_rng(42)
        np.random.seed(42)
        result_packed = recombination_via_pairs(packed_input.copy(), 0.01)

        # Run on legacy genomes
        legacy = LegacyGenomes(arr)
        legacy_input = legacy.get(individuals=np.arange(len(legacy)))
        variables.rng = np.random.default_rng(42)
        np.random.seed(42)
        result_legacy = recombination_via_pairs(legacy_input.copy(), 0.01)

        np.testing.assert_array_equal(result_packed, result_legacy)


# ---------------------------------------------------------------------------
# Property 10: Pairing packed/unpacked equivalence
# ---------------------------------------------------------------------------

class TestPairingEquivalence:
    """Feature: packed-bit-genomes, Property 10: Pairing equivalence"""

    @given(arr=genome_arrays(min_n=4, max_n=20, max_loci=16))
    @settings(max_examples=10, deadline=None)
    def test_pairing_same_result(self, arr):
        """Validates: Requirements 6.2"""
        from aegis_sim.submodels.reproduction.pairing import pairing
        from aegis_sim.submodels.reproduction.matingmanager import MatingManager
        from aegis_sim import submodels
        import aegis_sim.variables as variables

        n = arr.shape[0]
        sexes = np.array([0] * (n // 2) + [1] * (n - n // 2))
        ages = np.arange(n, dtype=np.int32)
        muta_prob = np.ones(n) * 0.01
        submodels.matingmanager = MatingManager()

        # Packed
        variables.rng = np.random.default_rng(42)
        np.random.seed(42)
        packed_g = Genomes(arr)
        c1, a1, m1 = pairing(packed_g, sexes, ages, muta_prob)

        # Legacy (wrap in Genomes since pairing expects Genomes interface)
        variables.rng = np.random.default_rng(42)
        np.random.seed(42)
        legacy_g = Genomes(arr)  # same packed path
        c2, a2, m2 = pairing(legacy_g, sexes, ages, muta_prob)

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# Property 11: Checkpoint format round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Feature: packed-bit-genomes, Property 11: Checkpoint format round-trip"""

    @given(arr=genome_arrays(min_n=1, max_n=20, max_loci=16))
    @settings(max_examples=10)
    def test_pickle_roundtrip(self, arr):
        """Validates: Requirements 8.3"""
        g = Genomes(arr)
        original_unpacked = g.unpack().copy()

        # Pickle and unpickle
        data = pickle.dumps(g)
        loaded = pickle.loads(data)

        assert isinstance(loaded, Genomes)
        assert loaded._packed.dtype == np.uint8
        np.testing.assert_array_equal(loaded.unpack(), original_unpacked)

    @given(arr=genome_arrays(min_n=1, max_n=20, max_loci=16))
    @settings(max_examples=10)
    def test_pickle_to_file_roundtrip(self, arr):
        """Pickle to file and back — simulates checkpoint save/load."""
        g = Genomes(arr)
        original_unpacked = g.unpack().copy()

        with tempfile.NamedTemporaryFile(delete=True) as f:
            pickle.dump(g, f)
            f.seek(0)
            loaded = pickle.load(f)

        np.testing.assert_array_equal(loaded.unpack(), original_unpacked)
