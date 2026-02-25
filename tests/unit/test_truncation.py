"""Unit tests for truncation helpers in RecordingManager."""

import pathlib
import pytest

from aegis_sim.recording.recordingmanager import RecordingManager


class TestCountRecordings:
    """Tests for _count_recordings (mirrors skip() logic)."""

    def test_rate_zero(self):
        assert RecordingManager._count_recordings(100, 0) == 0

    def test_negative_rate(self):
        assert RecordingManager._count_recordings(100, -1) == 0

    def test_step_zero(self):
        assert RecordingManager._count_recordings(0, 10) == 0

    def test_step_one_any_rate(self):
        # Step 1 always records
        assert RecordingManager._count_recordings(1, 10) == 1
        assert RecordingManager._count_recordings(1, 1) == 1
        assert RecordingManager._count_recordings(1, 100) == 1

    def test_rate_one(self):
        # Every step records: steps 1,2,3,4,5 = 5
        assert RecordingManager._count_recordings(5, 1) == 5

    def test_rate_ten(self):
        # step 1, 10, 20 = 3
        assert RecordingManager._count_recordings(25, 10) == 3
        # step 1, 10 = 2
        assert RecordingManager._count_recordings(10, 10) == 2
        # step 1 only
        assert RecordingManager._count_recordings(9, 10) == 1

    def test_rate_equals_steps(self):
        # step 1, 50 = 2
        assert RecordingManager._count_recordings(50, 50) == 2


class TestTruncateFile:
    """Tests for _truncate_file."""

    def test_truncate_removes_extra_lines(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a\nb\nc\nd\ne\n")
        RecordingManager._truncate_file(f, keep_lines=3)
        assert f.read_text() == "a\nb\nc\n"

    def test_truncate_noop_if_fewer_lines(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a\nb\n")
        RecordingManager._truncate_file(f, keep_lines=5)
        assert f.read_text() == "a\nb\n"

    def test_truncate_missing_file(self, tmp_path):
        # Should not raise
        RecordingManager._truncate_file(tmp_path / "nonexistent.csv", keep_lines=3)

    def test_truncate_to_zero(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a\nb\nc\n")
        RecordingManager._truncate_file(f, keep_lines=0)
        assert f.read_text() == ""
