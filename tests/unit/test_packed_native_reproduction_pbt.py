"""Property-based tests for packed-native reproduction pipeline.

Validates that the packed-native kernels (recombination, pairing, mutation)
produce bit-identical results to the bool-based implementations.
"""

import numpy as np
import pytest

from hypothesis import given, settings, assume
from hypothesis import strategies as st

import aegis_sim.variables as variables
from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim.submodels.reproduction.recombination import (
    recombination_via_pairs,
    recombination_via_pairs_packed,
)
from aegis_sim.submodels.reproduction.pairing import pairing, pairing_packed
from aegis_sim.submodels.reproduction.mutation import Mutator
from aegis_sim.submodels.reproduction.matingmanager import MatingManager
from aegis_sim import submodels


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def genome_4d(draw, min_n=4, max_n=40, max_loci=32):
    """Generate random 4D bool genome arrays (divisible-by-8 total bits)."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    bpl = 1  # modifying architecture uses BPL=1
    n_loci = draw(st.integers(min_value=8, max_value=max_loci)) // 8 * 8
    assume(n_loci >= 8)
    seed = draw(st.integers(0, 2**32 - 1))
    arr = np.random.default_rng(seed).integers(0, 2, size=(n, 2, n_loci, bpl)).astype(np.bool_)
    return arr, n_loci, bpl


# ---------------------------------------------------------------------------
# Property 2: Packed recombination equivalence
# ---------------------------------------------------------------------------

class TestPackedRecombinationEquivalence:
    """Feature: packed-native-reproduction, Property 2"""

    @given(data=genome_4d(min_n=4, max_n=30))
    @settings(max_examples=20, deadline=None)
    def test_recombination_packed_matches_bool(self, data):
        """Validates: Requirements 2.4"""
        arr, n_loci, bpl = data
        rate = 0.01

        # Bool path
        variables.rng = np.random.default_rng(77)
        np.random.seed(77)
        flat_bool = arr.reshape(len(arr), 2, -1).copy()
        result_bool = recombination_via_pairs(arr.copy(), rate)

        # Packed path
        variables.rng = np.random.default_rng(77)
        np.random.seed(77)
        g = Genomes(arr)
        packed = g.get_packed(np.arange(len(arr)))
        result_packed = recombination_via_pairs_packed(packed, n_loci, bpl, rate)

        # Unpack and compare
        total_bits = n_loci * bpl
        unpacked = np.unpackbits(result_packed, axis=-1, bitorder='big')[:, :, :total_bits]
        result_packed_bool = unpacked.reshape(len(arr), 2, n_loci, bpl).view(np.bool_)

        np.testing.assert_array_equal(result_bool, result_packed_bool)


# ---------------------------------------------------------------------------
# Property 6: Packed pairing equivalence
# ---------------------------------------------------------------------------

class TestPackedPairingEquivalence:
    """Feature: packed-native-reproduction, Property 6"""

    @given(data=genome_4d(min_n=4, max_n=30))
    @settings(max_examples=20, deadline=None)
    def test_pairing_packed_matches_bool(self, data):
        """Validates: Requirements 5.2"""
        arr, n_loci, bpl = data
        n = len(arr)
        sexes = np.array([0] * (n // 2) + [1] * (n - n // 2))
        ages = np.arange(n, dtype=np.int32)
        muta_prob = np.ones(n) * 0.01
        submodels.matingmanager = MatingManager()

        # Bool path
        variables.rng = np.random.default_rng(88)
        np.random.seed(88)
        c_bool, a_bool, m_bool = pairing(Genomes(arr), sexes, ages, muta_prob)

        # Packed path
        variables.rng = np.random.default_rng(88)
        np.random.seed(88)
        g = Genomes(arr)
        packed = g._packed.copy()
        c_packed, a_packed, m_packed = pairing_packed(packed, sexes, ages, muta_prob, n_loci, bpl)

        # Unpack packed children and compare
        total_bits = n_loci * bpl
        unpacked = np.unpackbits(c_packed, axis=-1, bitorder='big')[:, :, :total_bits]
        c_packed_bool = unpacked.reshape(len(c_packed), 2, n_loci, bpl).view(np.bool_)

        np.testing.assert_array_equal(c_bool, c_packed_bool)
        np.testing.assert_array_equal(a_bool, a_packed)
        np.testing.assert_array_equal(m_bool, m_packed)


# ---------------------------------------------------------------------------
# Property 4: Packed mutation by_index equivalence
# ---------------------------------------------------------------------------

class TestPackedMutationByIndexEquivalence:
    """Feature: packed-native-reproduction, Property 4"""

    @given(data=genome_4d(min_n=2, max_n=20, max_loci=24))
    @settings(max_examples=20, deadline=None)
    def test_mutation_packed_matches_bool(self, data):
        """Validates: Requirements 3.4"""
        arr, n_loci, bpl = data
        n = len(arr)
        muta_prob = np.ones(n) * 0.005
        ages = np.arange(n, dtype=np.int32)

        mut = Mutator()
        mut.init(MUTATION_RATIO=1, MUTATION_METHOD="by_index", MUTATION_AGE_MULTIPLIER=0)

        # Bool path
        variables.rng = np.random.default_rng(99)
        bool_genomes = arr.copy()
        result_bool = mut._mutate_by_index(bool_genomes, muta_prob.copy(), ages.copy())

        # Packed path
        variables.rng = np.random.default_rng(99)
        g = Genomes(arr)
        packed = g._packed.copy()
        result_packed = mut._mutate_by_index_packed(packed, muta_prob.copy(), ages.copy(), n_loci, bpl)

        # Unpack and compare
        total_bits = n_loci * bpl
        unpacked = np.unpackbits(result_packed, axis=-1, bitorder='big')[:, :, :total_bits]
        result_packed_bool = unpacked.reshape(n, 2, n_loci, bpl).view(np.bool_)

        np.testing.assert_array_equal(result_bool, result_packed_bool)


# ---------------------------------------------------------------------------
# Property 3: XOR bit isolation
# ---------------------------------------------------------------------------

class TestXORBitIsolation:
    """Feature: packed-native-reproduction, Property 3"""

    @given(data=genome_4d(min_n=1, max_n=5, max_loci=16))
    @settings(max_examples=20)
    def test_xor_flips_exactly_one_bit(self, data):
        """Validates: Requirements 3.2"""
        arr, n_loci, bpl = data
        g = Genomes(arr)
        packed = g._packed.copy()

        rng = np.random.default_rng(42)
        ind = rng.integers(0, len(arr))
        chrom = rng.integers(0, 2)
        locus = rng.integers(0, n_loci)
        bit = rng.integers(0, bpl)

        flat_bit = locus * bpl + bit
        byte_idx = flat_bit // 8
        bit_pos = flat_bit % 8

        before = packed.copy()
        packed[ind, chrom, byte_idx] ^= np.uint8(1 << (7 - bit_pos))

        # Unpack both and compare
        total_bits = n_loci * bpl
        before_bool = np.unpackbits(before, axis=-1, bitorder='big')[:, :, :total_bits]
        after_bool = np.unpackbits(packed, axis=-1, bitorder='big')[:, :, :total_bits]

        diff = before_bool != after_bool
        assert diff.sum() == 1


# ---------------------------------------------------------------------------
# Property 7: End-to-end pipeline equivalence
# ---------------------------------------------------------------------------

class TestEndToEndPipelineEquivalence:
    """Feature: packed-native-reproduction, Property 7"""

    @given(data=genome_4d(min_n=6, max_n=20, max_loci=24))
    @settings(max_examples=10, deadline=None)
    def test_full_pipeline_packed_matches_bool(self, data):
        """Validates: Requirements 7.2

        Runs the full reproduction pipeline (recombination → pairing → mutation)
        with both packed and bool paths using the same seed.
        """
        arr, n_loci, bpl = data
        n = len(arr)
        sexes = np.array([0] * (n // 2) + [1] * (n - n // 2))
        ages = np.arange(n, dtype=np.int32)
        muta_prob = np.ones(n) * 0.005
        rate = 0.01
        submodels.matingmanager = MatingManager()

        mut = Mutator()
        mut.init(MUTATION_RATIO=1, MUTATION_METHOD="by_index", MUTATION_AGE_MULTIPLIER=0)

        # Bool pipeline
        variables.rng = np.random.default_rng(55)
        np.random.seed(55)
        g_bool = arr.copy()
        g_bool = recombination_via_pairs(g_bool, rate)
        g_bool, ages_b, muta_b = pairing(Genomes(g_bool), sexes, ages, muta_prob)
        g_bool = mut._mutate_by_index(g_bool, muta_b, ages_b)

        # Packed pipeline
        variables.rng = np.random.default_rng(55)
        np.random.seed(55)
        g = Genomes(arr)
        packed = g._packed.copy()
        packed = recombination_via_pairs_packed(packed, n_loci, bpl, rate)
        packed, ages_p, muta_p = pairing_packed(packed, sexes, ages, muta_prob, n_loci, bpl)
        packed = mut._mutate_by_index_packed(packed, muta_p, ages_p, n_loci, bpl)

        # Unpack and compare
        total_bits = n_loci * bpl
        unpacked = np.unpackbits(packed, axis=-1, bitorder='big')[:, :, :total_bits]
        result_packed_bool = unpacked.reshape(len(packed), 2, n_loci, bpl).view(np.bool_)

        np.testing.assert_array_equal(g_bool, result_packed_bool)
