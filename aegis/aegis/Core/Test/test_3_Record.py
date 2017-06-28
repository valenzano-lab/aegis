from aegis.Core import Infodict, Config, Population, Record
import pytest,importlib,types,random,copy,string
import numpy as np

##############
## FIXTURES ##
##############

from test_1_Config import conf
from test_2a_Population_init import pop

@pytest.fixture()
def rec(request, conf):
    """Create a sample population from the default configuration."""
    return Record(conf)

###########
## TESTS ##
###########

class TestRecord:
    """Test Record object initialisation and methods."""

    def test_record_init(self, conf, rec):
        R = copy.deepcopy(rec)
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

    def test_p_calc(self,rec):
        """Test that p_calc returns the correct results under trivial and
        non-trivial cases, and returns value errors when appropriate."""
        bound = sorted([random.random(), random.random()])
        size0 = random.randint(10,50)
        size1 = random.randint(10,50)
        maxval = rec["n_states"]-1
        # Simple 1D arrays
        assert np.array_equal(rec.p_calc(np.zeros(size0), bound),
                np.tile(bound[0], size0))
        assert np.array_equal(rec.p_calc(np.tile(maxval, size0), bound),
                np.tile(bound[1], size0))
        # Simple 2D arrays
        assert np.array_equal(rec.p_calc(np.zeros([size0,size1]), bound),
                np.tile(bound[0], [size0,size1]))
        assert np.array_equal(rec.p_calc(np.tile(maxval,[size0,size1]), bound),
                np.tile(bound[1], [size0,size1]))
        # Random 2D arrays
        inval = random.choice(xrange(rec["n_states"]))
        inarray = np.tile(inval,[size0,size1])
        outarray = rec.p_calc(inarray, bound)
        exparray = bound[0] + (bound[1]-bound[0])*inarray/maxval
        print inarray.shape, outarray.shape, exparray.shape
        assert np.allclose(outarray, exparray)
        # Failure modes
        with pytest.raises(ValueError):
            rec.p_calc(np.tile(random.randint(-1,-100), [size0, size1]), bound)
        with pytest.raises(ValueError):
            rec.p_calc(np.tile(maxval+random.randint(1,100), [size0, size1]), bound)

    def test_p_surv_repr(self, rec):
        """Test that p_surv and p_repr return results equivalent to calling
        p_calc with the appropriate value ranges."""
        size0 = random.randint(10,50)
        size1 = random.randint(10,50)
        inval = random.choice(xrange(rec["n_states"]))
        inarray = np.tile(inval,[size0,size1])
        assert np.array_equal(rec.p_surv(inarray),
                rec.p_calc(inarray, rec["surv_bound"]))
        assert np.array_equal(rec.p_repr(inarray),
                rec.p_calc(inarray, rec["repr_bound"]))
