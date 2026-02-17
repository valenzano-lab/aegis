"""Unit tests for SimpleProgressRecorder.

Covers: check_when_last_updated (file modification time check).
"""

import time
import pytest

from aegis_sim.recording.simpleprogressrecorder import SimpleProgressRecorder


class TestCheckWhenLastUpdated:
    """Verify check_when_last_updated returns seconds since file modification."""

    def test_recently_written_file(self, tmp_path):
        """A just-written file reports near-zero seconds."""
        f = tmp_path / "test.log"
        f.write_text("hello")
        elapsed = SimpleProgressRecorder.check_when_last_updated(f)
        assert elapsed < 5

    def test_returns_positive_number(self, tmp_path):
        """Return value is a non-negative float."""
        f = tmp_path / "test.log"
        f.write_text("hello")
        elapsed = SimpleProgressRecorder.check_when_last_updated(f)
        assert isinstance(elapsed, float)
        assert elapsed >= 0
