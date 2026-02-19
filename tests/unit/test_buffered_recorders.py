"""Unit tests for buffered I/O in PopsizeRecorder and ResourcesRecorder.

Covers:
- Writes are buffered (not immediately on disk)
- flush_all writes buffered data to disk
- Auto-flush triggers at buffer limit (100 entries)
- Multiple files are tracked independently
- Flush is idempotent (double flush doesn't duplicate)
"""

import pytest
from unittest.mock import patch

from aegis_sim.recording.popsizerecorder import PopsizeRecorder
from aegis_sim.recording.resourcerecorder import ResourcesRecorder


class TestPopsizeRecorderBuffering:

    def test_write_buffers_in_memory(self, tmp_path):
        """A single write should not create the file until flush."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write(42, "test.csv")
        assert not (tmp_path / "test.csv").exists()

    def test_flush_writes_to_disk(self, tmp_path):
        """flush_all should write all buffered data."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write(10, "test.csv")
        rec.write(20, "test.csv")
        rec.flush_all()
        content = (tmp_path / "test.csv").read_text()
        assert content == "10\n20\n"

    def test_auto_flush_at_buffer_limit(self, tmp_path):
        """Buffer should auto-flush when it reaches 100 entries."""
        rec = PopsizeRecorder(odir=tmp_path)
        for i in range(100):
            rec.write(i, "test.csv")
        # After 100 writes, auto-flush should have triggered
        assert (tmp_path / "test.csv").exists()
        lines = (tmp_path / "test.csv").read_text().strip().splitlines()
        assert len(lines) == 100

    def test_multiple_files_independent(self, tmp_path):
        """Buffers for different files are independent."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write(1, "a.csv")
        rec.write(2, "b.csv")
        rec.write(3, "a.csv")
        rec.flush_all()

        assert (tmp_path / "a.csv").read_text() == "1\n3\n"
        assert (tmp_path / "b.csv").read_text() == "2\n"

    def test_double_flush_no_duplicates(self, tmp_path):
        """Flushing twice should not duplicate data."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write(42, "test.csv")
        rec.flush_all()
        rec.flush_all()
        content = (tmp_path / "test.csv").read_text()
        assert content == "42\n"

    def test_flush_then_write_appends(self, tmp_path):
        """Writes after a flush should append correctly."""
        rec = PopsizeRecorder(odir=tmp_path)
        rec.write(1, "test.csv")
        rec.flush_all()
        rec.write(2, "test.csv")
        rec.flush_all()
        lines = (tmp_path / "test.csv").read_text().strip().splitlines()
        assert lines == ["1", "2"]


class TestResourcesRecorderBuffering:

    def test_write_buffers_in_memory(self, tmp_path):
        """A single write should not create the file until flush."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            mock_res.capacity = 500.0
            rec.write_before_scavenging()
        assert not (tmp_path / "resources_before_scavenging.csv").exists()

    def test_flush_writes_to_disk(self, tmp_path):
        """flush_all should write all buffered data."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            mock_res.capacity = 100
            rec.write_before_scavenging()
            mock_res.capacity = 200
            rec.write_after_scavenging()
        rec.flush_all()

        assert (tmp_path / "resources_before_scavenging.csv").read_text() == "100\n"
        assert (tmp_path / "resources_after_scavenging.csv").read_text() == "200\n"

    def test_auto_flush_at_buffer_limit(self, tmp_path):
        """Buffer should auto-flush when it reaches 100 entries."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            for i in range(100):
                mock_res.capacity = float(i)
                rec.write_before_scavenging()
        assert (tmp_path / "resources_before_scavenging.csv").exists()
        lines = (tmp_path / "resources_before_scavenging.csv").read_text().strip().splitlines()
        assert len(lines) == 100

    def test_double_flush_no_duplicates(self, tmp_path):
        """Flushing twice should not duplicate data."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            mock_res.capacity = 42
            rec.write_before_scavenging()
        rec.flush_all()
        rec.flush_all()
        content = (tmp_path / "resources_before_scavenging.csv").read_text()
        assert content == "42\n"
