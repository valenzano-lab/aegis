"""Unit tests for FeatherRecorder, specifically empty population handling."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from aegis_sim.recording.featherrecorder import FeatherRecorder


class TestWriteGenotypesEmptyPopulation:

    def test_empty_population_writes_empty_feather(self, tmp_path):
        """write_genotypes should produce an empty feather file when population is empty."""
        rec = FeatherRecorder(odir=tmp_path)

        population = MagicMock()
        population.__len__ = lambda self: 0

        rec.write_genotypes(step=100, population=population)

        path = tmp_path / "snapshots" / "genotypes" / "100.feather"
        assert path.exists()
        df = pd.read_feather(path)
        assert len(df) == 0

    def test_nonempty_population_writes_genotypes(self, tmp_path):
        """write_genotypes should write genome data for a non-empty population."""
        rec = FeatherRecorder(odir=tmp_path)

        genomes = MagicMock()
        genomes.flatten.return_value = np.array([[1, 0, 1], [0, 1, 0]])

        population = MagicMock()
        population.__len__ = lambda self: 2
        population.genomes = genomes

        rec.write_genotypes(step=200, population=population)

        path = tmp_path / "snapshots" / "genotypes" / "200.feather"
        assert path.exists()
        df = pd.read_feather(path)
        assert len(df) == 2
