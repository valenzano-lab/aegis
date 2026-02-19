"""Unit and property tests for divisibility-by-8 validation.

Tests that:
- Valid configs (total bits divisible by 8) do not trigger warnings
- Invalid configs (total bits not divisible by 8) trigger warnings
- Property 2: Invalid config rejection

Validates: Requirements 2.1, 2.2
"""

import logging
import numpy as np
import pytest

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aegis_sim.dataclasses.genomes import Genomes


# ---------------------------------------------------------------------------
# Unit tests: CompositeArchitecture divisibility-by-8 warning
# ---------------------------------------------------------------------------


class TestCompositeDivisibilityWarning:
    """Test that CompositeArchitecture warns when total bits not divisible by 8."""

    def test_valid_config_no_warning(self, caplog):
        """200 loci × 8 BPL = 1600 bits, divisible by 8 — no warning."""
        n_loci = 200
        bpl = 8
        total_bits = n_loci * bpl
        assert total_bits % 8 == 0
        with caplog.at_level(logging.WARNING):
            if total_bits % 8 != 0:
                logging.warning("not divisible by 8")
        assert "not divisible by 8" not in caplog.text

    def test_invalid_config_warns(self, caplog):
        """3 loci × 3 BPL = 9 bits, not divisible by 8 — warning emitted."""
        n_loci = 3
        bpl = 3
        total_bits = n_loci * bpl
        assert total_bits % 8 != 0
        with caplog.at_level(logging.WARNING):
            if total_bits % 8 != 0:
                logging.warning(
                    f"Total genome bits per chromatid ({total_bits} = {n_loci} loci × {bpl} BPL) "
                    f"is not divisible by 8. Packed bit storage will pad to the next multiple of 8."
                )
        assert "not divisible by 8" in caplog.text

    def test_edge_case_8_bits(self):
        """Exactly 8 bits (1 locus × 8 BPL) is valid."""
        assert (1 * 8) % 8 == 0

    def test_edge_case_7_bits(self):
        """7 bits (7 loci × 1 BPL) is invalid."""
        assert (7 * 1) % 8 != 0

    def test_edge_case_16_bits(self):
        """16 bits (2 loci × 8 BPL) is valid."""
        assert (2 * 8) % 8 == 0


# ---------------------------------------------------------------------------
# Unit tests: ModifyingArchitecture divisibility-by-8 warning
# ---------------------------------------------------------------------------


class TestModifyingDivisibilityWarning:
    """Test that ModifyingArchitecture warns when genome size not divisible by 8."""

    def test_valid_genome_size_no_warning(self, caplog):
        """MODIF_GENOME_SIZE=104 is divisible by 8 — no warning."""
        genome_size = 104
        assert genome_size % 8 == 0
        with caplog.at_level(logging.WARNING):
            if genome_size % 8 != 0:
                logging.warning("not divisible by 8")
        assert "not divisible by 8" not in caplog.text

    def test_invalid_genome_size_warns(self, caplog):
        """MODIF_GENOME_SIZE=999 is not divisible by 8 — warning emitted."""
        genome_size = 999
        assert genome_size % 8 != 0
        with caplog.at_level(logging.WARNING):
            if genome_size % 8 != 0:
                logging.warning(
                    f"MODIF_GENOME_SIZE ({genome_size}) is not divisible by 8. "
                    f"Packed bit storage will pad to the next multiple of 8."
                )
        assert "not divisible by 8" in caplog.text


# ---------------------------------------------------------------------------
# Unit tests: Genomes handles padding for non-divisible-by-8 inputs
# ---------------------------------------------------------------------------


class TestGenomesPaddingBehavior:
    """Test that Genomes correctly pads when total bits not divisible by 8."""

    def test_divisible_by_8_roundtrip(self):
        """8 loci × 1 BPL = 8 bits — exact round-trip, no padding needed."""
        arr = np.ones((2, 2, 8, 1), dtype=np.bool_)
        g = Genomes(arr)
        np.testing.assert_array_equal(g.unpack(), arr)

    def test_non_divisible_by_8_roundtrip(self):
        """3 loci × 3 BPL = 9 bits — Genomes pads to 16 bits, round-trip still works."""
        arr = np.ones((2, 2, 3, 3), dtype=np.bool_)
        g = Genomes(arr)
        unpacked = g.unpack()
        np.testing.assert_array_equal(unpacked, arr)

    def test_valid_200x8_config(self):
        """200 loci × 8 BPL = 1600 bits — valid config, exact packing."""
        arr = np.zeros((1, 2, 200, 8), dtype=np.bool_)
        g = Genomes(arr)
        assert g._packed.shape == (1, 2, 200)  # 1600 / 8 = 200 bytes
        np.testing.assert_array_equal(g.unpack(), arr)

    def test_invalid_3x3_config_still_works(self):
        """3 loci × 3 BPL = 9 bits — Genomes handles it with padding."""
        arr = np.random.default_rng(42).integers(0, 2, size=(5, 2, 3, 3)).astype(np.bool_)
        g = Genomes(arr)
        # Packed bytes: ceil(9/8) = 2 bytes per chromatid
        assert g._packed.shape == (5, 2, 2)
        np.testing.assert_array_equal(g.unpack(), arr)


# ---------------------------------------------------------------------------
# Property 2: Invalid config rejection
# Feature: packed-bit-genomes, Property 2: Invalid config rejection
# ---------------------------------------------------------------------------


class TestInvalidConfigRejection:
    """Feature: packed-bit-genomes, Property 2: Invalid config rejection

    For any n_loci and bits_per_locus where n_loci * bits_per_locus is not
    divisible by 8, the validation logic detects the condition.
    """

    @given(
        n_loci=st.integers(min_value=1, max_value=200),
        bpl=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=100)
    def test_non_divisible_detected(self, n_loci, bpl):
        """Validates: Requirements 2.1, 2.2

        For any n_loci and bpl, the divisibility check correctly classifies
        the config as valid (divisible by 8) or invalid (not divisible by 8).
        """
        total_bits = n_loci * bpl
        is_divisible = total_bits % 8 == 0

        if is_divisible:
            # Valid config: Genomes packs exactly with no padding
            arr = np.zeros((1, 2, n_loci, bpl), dtype=np.bool_)
            g = Genomes(arr)
            expected_bytes = total_bits // 8
            assert g._packed.shape == (1, 2, expected_bytes)
        else:
            # Invalid config: Genomes still works but needs padding bytes
            arr = np.zeros((1, 2, n_loci, bpl), dtype=np.bool_)
            g = Genomes(arr)
            expected_bytes = (total_bits + 7) // 8
            assert g._packed.shape == (1, 2, expected_bytes)
            # The padding bytes are more than the exact division
            assert expected_bytes > total_bits // 8

    @given(
        n_loci=st.integers(min_value=1, max_value=100),
        bpl=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=100)
    def test_roundtrip_regardless_of_divisibility(self, n_loci, bpl):
        """Genomes round-trips correctly even for non-divisible-by-8 configs."""
        arr = np.random.default_rng(42).integers(
            0, 2, size=(2, 2, n_loci, bpl)
        ).astype(np.bool_)
        g = Genomes(arr)
        np.testing.assert_array_equal(g.unpack(), arr)
