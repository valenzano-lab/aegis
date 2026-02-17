"""Unit tests for ResourcesRecorder.

Covers: write_before_scavenging and write_after_scavenging by patching
the module-level resources.capacity global.
"""

import pytest
from unittest.mock import patch

from aegis_sim.recording.resourcerecorder import ResourcesRecorder


class TestWriteBeforeScavenging:
    """Verify write_before_scavenging appends capacity to CSV."""

    def test_writes_capacity_value(self, tmp_path):
        """Written line matches the current resources.capacity."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            mock_res.capacity = 500.0
            rec.write_before_scavenging()
        content = (tmp_path / "resources_before_scavenging.csv").read_text()
        assert content == "500.0\n"

    def test_appends_multiple(self, tmp_path):
        """Multiple writes append one line each."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            mock_res.capacity = 100
            rec.write_before_scavenging()
            mock_res.capacity = 80
            rec.write_before_scavenging()
        lines = (tmp_path / "resources_before_scavenging.csv").read_text().strip().splitlines()
        assert lines == ["100", "80"]

    def test_integer_capacity(self, tmp_path):
        """Integer capacity is written without decimal."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            mock_res.capacity = 1000
            rec.write_before_scavenging()
        content = (tmp_path / "resources_before_scavenging.csv").read_text()
        assert content == "1000\n"


class TestWriteAfterScavenging:
    """Verify write_after_scavenging appends capacity to CSV."""

    def test_writes_capacity_value(self, tmp_path):
        """Written line matches the current resources.capacity."""
        rec = ResourcesRecorder(odir=tmp_path)
        with patch("aegis_sim.recording.resourcerecorder.resources") as mock_res:
            mock_res.capacity = 350.5
            rec.write_after_scavenging()
        content = (tmp_path / "resources_after_scavenging.csv").read_text()
        assert content == "350.5\n"
