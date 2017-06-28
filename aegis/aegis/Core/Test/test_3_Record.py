from aegis.Core import Infodict, Config, Population, Record
import pytest,importlib,types,random,copy,string
import numpy as np

from test_1_Config import conf 

class TestRecord:
    """Test Record object initialisation and methods."""

    def test_record_init(self, conf):
        R = Record(conf)
        # Conf param entries should be inherited in Record
        for k in conf.keys():
            assert np.array_equal(R[k], np.array(conf[k]))
            assert R.get_info(k) == conf.get_info(k)
        # R and V successfully renamed
        assert np.array_equal(R["res_regen_constant"], R["R"])
        assert np.array_equal(R["res_regen_prop"], R["V"])
        assert R.get_info("res_regen_constant") == R.get_info("R")
        assert R.get_info("res_regen_prop") == R.get_info("V")
        # Check basic run info
        assert np.array_equal(R["dieoff"], np.array(False))
        assert np.array_equal(R["prev_failed"], np.array(0))
        assert np.array_equal(R["percent_dieoff"], np.array(0))
        # Per-stage data entry
        a0 = np.zeros(R["number_of_stages"])
        a1 = np.zeros([R["number_of_stages"],R["max_ls"]])
        assert np.array_equal(R["population_size"], a0)
        assert np.array_equal(R["resources"], a0)
        assert np.array_equal(R["surv_penf"], a0)
        assert np.array_equal(R["repr_penf"], a0)
        assert np.array_equal(R["age_distribution"], a1)
        # Snapshot population placeholders
        assert R["snapshot_pops"] == [0]*R["number_of_snapshots"]
        # Empty names for final computation
        def iszero(*names):
            for name in names: assert R[name] == 0
        # Genotype sum statistics (density and average)
        iszero("density_per_locus", "density", "mean_gt", "var_gt", "entropy_gt")
        # Survival and reproduction
        iszero("cmv_surv", "junk_cmv_surv", "prob_mean", "prob_var", "junk_mean",
                "junk_var", "fitness_term", "junk_fitness_term", "repr_value",
                "junk_repr_value")
        # Per-bit statistics, actual death
        iszero("n1", "n1_var", "entropy_bits", "actual_death_rate")
        # Sliding windows
        iszero("population_size_window_mean", "population_size_window_var",
                "resources_window_mean", "resources_window_var",
                "n1_window_mean", "n1_window_var")


