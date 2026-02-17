"""Unit tests for PopsizeRecorder.

Covers: write() CSV append, write_egg_num_after_reproduction (None eggs).
"""

import pytest

from aegis_sim.recording.popsizerecorder import PopsizeRecorder


class TestPopsizeWrite:
    """Verify write() appends a single integer line to a CSV file."""

    def test_write_creates_file(self, tmp_path):
        """First write creates the file with one line."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write(42, "test.csv")
        content = (tmp_path / "test.csv").read_text()
        assert content == "42\n"

    def test_write_appends(self, tmp_path):
        """Multiple writes append lines."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write(10, "test.csv")
        rec.write(20, "test.csv")
        rec.write(30, "test.csv")
        lines = (tmp_path / "test.csv").read_text().strip().splitlines()
        assert lines == ["10", "20", "30"]


class TestEggNumAfterReproduction:
    """Verify write_egg_num_after_reproduction handles None eggs."""

    def test_none_eggs_writes_zero(self, tmp_path):
        """When eggs is None, writes 0."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write_egg_num_after_reproduction(eggs=None)
        content = (tmp_path / "eggnum_after_reproduction.csv").read_text()
        assert content == "0\n"
