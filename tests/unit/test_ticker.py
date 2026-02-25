"""Unit tests for the Ticker recorder.

Covers: write() file creation and timestamp format, read() for existing
and missing files, since_last() elapsed time calculation, and
has_stopped() liveness check.
"""

import time
import pytest

from aegis_sim.recording.ticker import Ticker


class TestTickerWrite:
    """Verify write() creates a ticker file with a valid timestamp."""

    def test_write_creates_file(self, tmp_path):
        """After write(), the ticker file exists on disk."""
        t = Ticker(TICKER_RATE=60, odir=tmp_path)
        t.write()
        assert t.ticker_path.exists()

    def test_write_contains_timestamp(self, tmp_path):
        """Written content is a 19-char YYYY-MM-DD HH:MM:SS timestamp."""
        t = Ticker(TICKER_RATE=60, odir=tmp_path)
        t.write()
        content = t.ticker_path.read_text()
        assert len(content) == 19
        assert content[4] == "-"
        assert content[10] == " "


class TestTickerRead:
    """Verify read() returns file content or None."""

    def test_read_returns_written_content(self, tmp_path):
        """After write(), read() returns the 19-char timestamp string."""
        t = Ticker(TICKER_RATE=60, odir=tmp_path)
        t.write()
        content = t.read()
        assert content is not None
        assert len(content) == 19

    def test_read_missing_file_returns_none(self, tmp_path):
        """Before any write(), read() returns None."""
        t = Ticker(TICKER_RATE=60, odir=tmp_path)
        result = t.read()
        assert result is None


class TestTickerSinceLast:
    """Verify since_last() computes seconds since the last write."""

    def test_since_last_is_small(self, tmp_path):
        """Immediately after write(), elapsed time is near zero."""
        t = Ticker(TICKER_RATE=60, odir=tmp_path)
        t.write()
        elapsed = t.since_last()
        assert elapsed is not None
        assert elapsed < 5

    def test_since_last_missing_file(self, tmp_path):
        """With no ticker file, since_last() returns None."""
        t = Ticker(TICKER_RATE=60, odir=tmp_path)
        result = t.since_last()
        assert result is None


class TestTickerHasStopped:
    """Verify has_stopped() liveness detection."""

    def test_just_written_not_stopped(self, tmp_path):
        """A freshly written ticker is not considered stopped."""
        t = Ticker(TICKER_RATE=60, odir=tmp_path)
        t.write()
        assert t.has_stopped() is False
