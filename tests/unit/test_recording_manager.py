"""Unit tests for RecordingManager static helpers.

Covers: make_odir (directory creation, overwrite behavior),
_count_recordings (skip-logic mirror), _truncate_file (line truncation).
"""

import pathlib
import pytest

from aegis_sim.recording.recordingmanager import RecordingManager


class TestMakeOdir:
    """Verify make_odir derives output path and handles overwrite."""

    def test_returns_stem_path(self, tmp_path):
        """Output dir is config path's parent / stem (no .yml)."""
        config = tmp_path / "my_sim.yml"
        config.touch()
        result = RecordingManager.make_odir(config, overwrite=False)
        assert result == tmp_path / "my_sim"

    def test_raises_if_exists_no_overwrite(self, tmp_path):
        """Existing output dir without overwrite raises Exception."""
        config = tmp_path / "my_sim.yml"
        config.touch()
        (tmp_path / "my_sim").mkdir()
        with pytest.raises(Exception, match="already exists"):
            RecordingManager.make_odir(config, overwrite=False)

    def test_overwrite_removes_existing(self, tmp_path):
        """With overwrite=True, existing output dir is removed."""
        config = tmp_path / "my_sim.yml"
        config.touch()
        odir = tmp_path / "my_sim"
        odir.mkdir()
        (odir / "old_file.txt").touch()
        result = RecordingManager.make_odir(config, overwrite=True)
        assert result == odir
        assert not odir.exists()

    def test_nonexistent_dir_ok(self, tmp_path):
        """When output dir doesn't exist, returns path without error."""
        config = tmp_path / "fresh_sim.yml"
        config.touch()
        result = RecordingManager.make_odir(config, overwrite=False)
        assert result == tmp_path / "fresh_sim"


class TestCountRecordings:
    """Verify _count_recordings mirrors the skip() logic.

    Rule: always record at step 1, then at every step divisible by rate.
    """

    def test_rate_zero(self):
        """Rate 0 (disabled) means no recordings."""
        assert RecordingManager._count_recordings(100, 0) == 0

    def test_negative_rate(self):
        """Negative rate (disabled) means no recordings."""
        assert RecordingManager._count_recordings(100, -1) == 0

    def test_step_zero(self):
        """Zero steps means no recordings."""
        assert RecordingManager._count_recordings(0, 10) == 0

    def test_step_one_any_rate(self):
        """Step 1 always records regardless of rate."""
        assert RecordingManager._count_recordings(1, 10) == 1
        assert RecordingManager._count_recordings(1, 1) == 1
        assert RecordingManager._count_recordings(1, 100) == 1

    def test_rate_one(self):
        """Rate 1 records every step: steps 1..5 = 5 recordings."""
        assert RecordingManager._count_recordings(5, 1) == 5

    def test_rate_ten(self):
        """Rate 10: step 1 + multiples of 10."""
        assert RecordingManager._count_recordings(25, 10) == 3   # 1, 10, 20
        assert RecordingManager._count_recordings(10, 10) == 2   # 1, 10
        assert RecordingManager._count_recordings(9, 10) == 1    # 1 only

    def test_rate_equals_steps(self):
        """Rate == steps: step 1 + step 50 = 2 recordings."""
        assert RecordingManager._count_recordings(50, 50) == 2


class TestTruncateFile:
    """Verify _truncate_file keeps only the first N lines."""

    def test_truncate_removes_extra_lines(self, tmp_path):
        """File with 5 lines truncated to 3 keeps first 3."""
        f = tmp_path / "data.csv"
        f.write_text("a\nb\nc\nd\ne\n")
        RecordingManager._truncate_file(f, keep_lines=3)
        assert f.read_text() == "a\nb\nc\n"

    def test_truncate_noop_if_fewer_lines(self, tmp_path):
        """File with fewer lines than keep_lines is unchanged."""
        f = tmp_path / "data.csv"
        f.write_text("a\nb\n")
        RecordingManager._truncate_file(f, keep_lines=5)
        assert f.read_text() == "a\nb\n"

    def test_truncate_missing_file(self, tmp_path):
        """Truncating a nonexistent file does not raise."""
        RecordingManager._truncate_file(tmp_path / "nonexistent.csv", keep_lines=3)

    def test_truncate_to_zero(self, tmp_path):
        """Truncating to 0 lines empties the file."""
        f = tmp_path / "data.csv"
        f.write_text("a\nb\nc\n")
        RecordingManager._truncate_file(f, keep_lines=0)
        assert f.read_text() == ""
