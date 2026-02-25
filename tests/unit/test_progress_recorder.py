"""Unit tests for ProgressRecorder.

Covers: get_dhm() static time formatting, init_headers() file creation,
write() progress log output with patched globals, and
write_to_progress_log() content formatting.
"""

import time
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from aegis_sim.recording.progressrecorder import ProgressRecorder


class TestGetDhm:
    """Verify get_dhm() converts seconds to 'd`HH:MM' format."""

    def test_zero_seconds(self):
        """0 seconds formats as 0 days, 00:00."""
        assert ProgressRecorder.get_dhm(0) == "0`00:00"

    def test_seconds_only(self):
        """30 seconds is still 0`00:00 (sub-minute truncated)."""
        assert ProgressRecorder.get_dhm(30) == "0`00:00"

    def test_one_minute(self):
        """60 seconds formats as 0`00:01."""
        assert ProgressRecorder.get_dhm(60) == "0`00:01"

    def test_one_hour(self):
        """3600 seconds formats as 0`01:00."""
        assert ProgressRecorder.get_dhm(3600) == "0`01:00"

    def test_one_day(self):
        """86400 seconds formats as 1`00:00."""
        assert ProgressRecorder.get_dhm(86400) == "1`00:00"

    def test_mixed(self):
        """1 day, 2 hours, 30 minutes = 95400 seconds."""
        seconds = 86400 + 2 * 3600 + 30 * 60
        assert ProgressRecorder.get_dhm(seconds) == "1`02:30"

    def test_multi_day(self):
        """3 days, 14 hours, 7 minutes."""
        seconds = 3 * 86400 + 14 * 3600 + 7 * 60
        assert ProgressRecorder.get_dhm(seconds) == "3`14:07"

    def test_fractional_seconds_truncated(self):
        """Fractional seconds are truncated, not rounded."""
        seconds = 3600 + 59.9  # 1 hour + 59.9 seconds (< 1 min)
        assert ProgressRecorder.get_dhm(seconds) == "0`01:00"


class TestInitHeaders:
    """Verify init_headers writes a header row to progress.log."""

    def test_creates_progress_log(self, tmp_path):
        """Constructor creates progress.log with a header line."""
        rec = ProgressRecorder(odir=tmp_path, resuming=False)
        log = (tmp_path / "progress.log").read_text()
        assert "step" in log
        assert "ETA" in log
        assert "popsize" in log

    def test_resuming_skips_header(self, tmp_path):
        """With resuming=True, no header is written."""
        rec = ProgressRecorder(odir=tmp_path, resuming=True)
        assert not (tmp_path / "progress.log").exists()


class TestWriteToProgressLog:
    """Verify write_to_progress_log appends a formatted row."""

    def test_appends_content(self, tmp_path):
        """A tuple of values is appended as a pipe-delimited row."""
        rec = ProgressRecorder(odir=tmp_path, resuming=True)
        # Create the file first
        (tmp_path / "progress.log").write_bytes(b"")
        rec.write_to_progress_log((10, "0`00:01", "0`00:05", "0`00:00", 600, 200))
        content = (tmp_path / "progress.log").read_text()
        assert "10" in content
        assert "200" in content


class TestWrite:
    """Verify write() produces a progress log entry with patched globals."""

    def test_write_appends_line(self, tmp_path):
        """With skip() returning False, write() appends a data row."""
        rec = ProgressRecorder(odir=tmp_path, resuming=False)
        rec.time_start = time.time() - 10  # pretend 10 seconds elapsed

        fake_params = SimpleNamespace(STEPS_PER_SIMULATION=100, LOGGING_RATE=10)
        fake_config_path = SimpleNamespace(stem="test_sim")

        with patch("aegis_sim.recording.progressrecorder.skip", return_value=False), \
             patch("aegis_sim.recording.progressrecorder.variables") as mock_vars, \
             patch("aegis_sim.recording.progressrecorder.parametermanager") as mock_pm:
            mock_vars.steps = 50
            mock_vars.custom_config_path = fake_config_path
            mock_pm.parameters = fake_params
            rec.write(popsize=200)

        lines = (tmp_path / "progress.log").read_text().strip().splitlines()
        # First line is header, second is data
        assert len(lines) == 2
        assert "200" in lines[1]
        assert "50" in lines[1]

    def test_write_skipped_when_skip_true(self, tmp_path):
        """When skip() returns True, no data row is written."""
        rec = ProgressRecorder(odir=tmp_path, resuming=False)

        with patch("aegis_sim.recording.progressrecorder.skip", return_value=True):
            rec.write(popsize=100)

        lines = (tmp_path / "progress.log").read_text().strip().splitlines()
        # Only the header line
        assert len(lines) == 1
