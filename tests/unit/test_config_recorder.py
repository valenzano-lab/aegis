"""Unit tests for ConfigRecorder.

Covers: write_final_config_file (YAML round-trip).
"""

import yaml
import pytest

from aegis_sim.recording.configrecorder import ConfigRecorder


class TestConfigRecorderWrite:
    """Verify write_final_config_file dumps valid YAML."""

    def test_writes_yaml(self, tmp_path):
        """Written file is valid YAML matching the input dict."""
        rec = ConfigRecorder(odir=tmp_path)
        config = {"STEPS_PER_SIMULATION": 100, "AGE_LIMIT": 50, "RANDOM_SEED": 42}
        rec.write_final_config_file(config)
        loaded = yaml.safe_load(rec.path.read_text())
        assert loaded == config

    def test_creates_file(self, tmp_path):
        """File is created at odir/final_config.yml."""
        rec = ConfigRecorder(odir=tmp_path)
        rec.write_final_config_file({"key": "value"})
        assert (tmp_path / "final_config.yml").exists()
