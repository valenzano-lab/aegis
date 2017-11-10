from aegis.Core import Infodict, Config, Population, Outpop, Record
import pytest,importlib,types,random,copy,string
import numpy as np

#########################
## AUXILIARY FUNCTIONS ##
#########################

def static_fill(rec_obj, pop_obj):
    """Fill a whole record object with the same initial
    population state."""
    r = copy.deepcopy(rec_obj)
    n,s,c = r["number_of_stages"], r["snapshot_stages"], 0
    for x in xrange(n):
        snapshot = c if x in s else -1
        r.update(pop_obj, 100, 1, 1, x, snapshot)
        c += 1 if x in s else 0
    return r

##############
## FIXTURES ##
##############

from test_1_Config import conf, conf_path
from test_2a_Population_init import pop

@pytest.fixture(scope="module")
def rec(request, conf):
    """Create a sample population from the default configuration."""
    return Record(conf)

@pytest.fixture(scope="module")
def pop1(request, pop):
    """Population of young adults with genomes filled with ones."""
    p = pop.clone()
    p.genomes = np.ones(p.genomes.shape).astype(int)
    p.ages = np.tile(p.maturity, p.N)
    p.generations = np.zeros(p.N, dtype=int)
    return p

@pytest.fixture(scope="module")
def rec1(request, rec, pop1):
    """Record filled with initial state of pop1."""
    return static_fill(rec, pop1)

@pytest.fixture(scope="module")
def rec2(request, rec1):
    """Record filled with initial state of pop1."""
    return copy.deepcopy(rec1)

###########
## TESTS ##
###########

class TestRecord:
    """Test Record object initialisation and methods."""

    ## INITIALISATION ##

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
        # Per-stage data entry
        a0 = np.zeros(R["number_of_stages"])
        a1 = np.zeros([R["number_of_stages"],R["max_ls"]])
        assert np.array_equal(R["population_size"], a0)
        assert np.array_equal(R["resources"], a0)
        assert np.array_equal(R["surv_penf"], a0)
        assert np.array_equal(R["repr_penf"], a0)
        assert np.array_equal(R["age_distribution"], a1)
        # Snapshot population placeholders
        assert R["snapshot_pops"] == [0]*R["number_of_snapshots"][0]
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

    ## PROBABILITIES ##

    def test_p_calc(self,rec):
        """Test that p_calc returns the correct results under trivial and
        non-trivial cases, and returns value errors when appropriate."""
        bound = sorted([random.random(), random.random()])
        size0 = random.randint(10,50)
        size1 = random.randint(10,50)
        maxval = rec["n_states"]-1
        # Simple 1D arrays
        assert np.allclose(rec.p_calc(np.zeros(size0), bound),
                np.tile(bound[0], size0))
        assert np.allclose(rec.p_calc(np.tile(maxval, size0), bound),
                np.tile(bound[1], size0))
        # Simple 2D arrays
        assert np.allclose(rec.p_calc(np.zeros([size0,size1]), bound),
                np.tile(bound[0], [size0,size1]))
        assert np.allclose(rec.p_calc(np.tile(maxval,[size0,size1]), bound),
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

    ## PER-STAGE RECORDING ##

    def test_update_quick(self, rec, pop):
        """Test that every-stage update function records correctly."""
        rec2 = copy.deepcopy(rec)
        r = rec2.get_value
        rec2.update(pop, 100, 1, 1, 0, -1)
        agedist=np.bincount(pop.ages,minlength=pop.max_ls)/float(pop.N)
        assert r("resources")[0] == 100
        assert r("population_size")[0] == pop.N
        assert r("surv_penf")[0] == 1
        assert r("repr_penf")[0] == 1
        assert np.array_equal(r("age_distribution")[0], agedist)
        for n in xrange(len(r("snapshot_pops"))):
            assert r("snapshot_pops")[n] == 0

    def test_update_full(self, rec, pop):
        """Test that snapshot update function records correctly."""
        rec2 = copy.deepcopy(rec)
        pop2 = pop.clone()
        np.random.shuffle(pop2.genmap)
        np.random.shuffle(pop2.ages)
        np.random.shuffle(pop2.genomes)
        rec2.update(pop2, 200, 2, 2, 0, 0)
        r = rec2.get_value
        agedist=np.bincount(pop2.ages,minlength=pop2.max_ls)/float(pop2.N)
        # Per-stage factors
        assert r("population_size")[0] == pop2.N
        assert r("resources")[0] == 200
        assert r("surv_penf")[0] == 2
        assert r("repr_penf")[0] == 2
        assert np.array_equal(r("age_distribution")[0], agedist)
        for n in xrange(1,len(r("snapshot_pops"))):
            assert r("snapshot_pops")[n] == 0
        # Snapshot population
        p = r("snapshot_pops")[0]
        assert isinstance(p, Outpop)
        assert np.array_equal(p.genmap, pop2.genmap)
        assert np.array_equal(p.ages, pop2.ages)
        assert np.array_equal(p.genomes, pop2.genomes)
        assert np.array_equal(p.generations, pop2.generations)

    ## FINALISATION ##

    def test_compute_locus_density(self, rec1, rec2):
        """Test that compute_locus_density performs correctly for a
        genome filled with 1's.""" #! TODO: How about in a normal case?
        rec1.compute_locus_density()
        m,l,b = rec1["number_of_snapshots"], rec1["max_ls"], rec1["n_states"]
        g,nn,mt = len(rec1["genmap"]), rec1["n_neutral"], rec1["maturity"]
        llist = ["a","n","r","s"]
        dims = {"a":[b,m,g],"n":[b,m,nn],"r":[b,m,l-mt], "s":[b,m,l]}
        obj = rec1["density_per_locus"]
        assert sorted(obj.keys()) == llist
        for l in llist:
            check = np.zeros(dims[l])
            check[-1] = 1
            assert np.array_equal(obj[l], check)

    def test_compute_total_density(self, rec1):
        """Test that compute_total_density performs correctly for a
        genome filled with 1's.""" #! TODO: How about in a normal case?
        rec1.compute_total_density()
        m,b = rec1["number_of_snapshots"], rec1["n_states"]
        llist = ["a","n","r","s"]
        obj = rec1["density"]
        assert sorted(obj.keys()) == llist
        for l in llist:
            check = np.zeros([b,m])
            check[-1] = 1
            assert np.array_equal(obj[l], check)

    def test_compute_genotype_mean_var(self, rec1):
        """Test that compute_genotype_mean_var performs correctly for a
        genome filled with 1's.""" #! TODO: How about in a normal case?
        rec1.compute_genotype_mean_var()
        m,l,b = rec1["number_of_snapshots"], rec1["max_ls"], rec1["n_states"]
        g,nn,mt = len(rec1["genmap"]), rec1["n_neutral"], rec1["maturity"]
        llist = ["a","n","r","s"]
        dims = {"a":[m,g],"n":[m,nn],"r":[m,l-mt],"s":[m,l]}
        for k in ["mean_gt","var_gt"]:
            obj = rec1[k]
            assert sorted(obj.keys()) == llist
            for l in llist:
                check = np.zeros(dims[l])
                if k == "mean_gt": check[:] = b-1 # All genotypes maximal
                assert np.array_equal(obj[l], check)

    def test_compute_surv_repr_probabilities_true(self, rec1):
        """Test that compute_surv_repr_probabilities_true performs 
        correctly for a genome filled with 1's.""" #! TODO: How about in a normal case?
        rec1.compute_surv_repr_probabilities_true()
        # Define parameters
        m,ls = rec1["number_of_snapshots"], rec1["max_ls"]
        ns,mt= rec1["n_states"], rec1["maturity"]
        llist = ["repr", "surv"]
        dims = {"surv":[m,ls], "repr":[m,ls-mt]}
        vmax = {"surv":rec1.p_surv(ns-1), "repr":rec1.p_repr(ns-1)}
        # Test vs expectation
        for k in ["mean", "var"]: assert sorted(rec1["prob_"+k].keys()) == llist
        for l in llist:
            assert np.array_equal(rec1["prob_mean"][l], np.tile(vmax[l], dims[l]))
            assert np.array_equal(rec1["prob_var"][l], np.zeros(dims[l]))

    def test_surv_repr_probabilities_junk(self, rec1):
        """Test that compute_surv_repr_probabilities_junk performs 
        correctly for a genome filled with 1's.""" #! TODO: How about in a normal case?
        rec1.compute_surv_repr_probabilities_junk()
        # Define parameters
        m,nn,ns = rec1["number_of_snapshots"],rec1["n_neutral"],rec1["n_states"]
        llist = ["repr", "surv"]
        vmax = {"surv":rec1.p_surv(ns-1), "repr":rec1.p_repr(ns-1)}
        # Test vs expectation
        for k in ["mean", "var"]: assert sorted(rec1["junk_"+k].keys()) == llist
        for l in llist:
            assert np.array_equal(rec1["junk_mean"][l], np.tile(vmax[l], [m,nn]))
            assert np.array_equal(rec1["junk_var"][l], np.zeros([m,nn]))

    def test_compute_cmv_surv(self, rec1):
        """Test that cumulative survival probabilities are computed
        correctly for a genome filled with 1's."""
        rec1.compute_cmv_surv()
        # Define parameters
        m,ls,ns = rec1["number_of_snapshots"], rec1["max_ls"], rec1["n_states"]
        # Test vs expectation
        cs = np.tile(rec1.p_surv(ns-1)**np.arange(ls), [m,1])
        assert np.allclose(rec1["cmv_surv"], cs)
        assert np.allclose(rec1["junk_cmv_surv"], cs)

    def test_compute_mean_repr(self, rec1):
        """Test that mean reproduction probability calculations are
        computed correctly for a genome filled with 1's."""
        rec1.compute_mean_repr()
        # Define parameters
        sex = rec1["repr_mode"] in ["sexual", "assort_only"]
        div = 2 if sex else 1
        m,ns = rec1["number_of_snapshots"], rec1["n_states"]
        ls,mt = rec1["max_ls"], rec1["maturity"]
        # Test vs expectation
        mr = np.tile(rec1.p_repr(ns-1), [m,ls])
        mr[:,:mt] = 0
        assert np.allclose(rec1["mean_repr"], mr/div)
        assert np.allclose(rec1["junk_repr"], mr/div)

    def test_compute_fitness(self, rec1):
        """Test that per-age and total fitness are computed correctly
        for a genome filled with 1's."""
        # Update record
        rec1.compute_fitness()
        # Test vs expectation
        assert np.allclose(rec1["fitness_term"], 
                rec1["cmv_surv"]*rec1["mean_repr"])
        assert np.allclose(rec1["junk_fitness_term"],
                rec1["junk_cmv_surv"]*rec1["junk_repr"])
        assert np.allclose(rec1["fitness"], np.sum(rec1["fitness_term"], 1))
        assert np.allclose(rec1["junk_fitness"],
                np.sum(rec1["junk_fitness_term"], 1))

    def test_compute_reproductive_value(self, rec1):
        # Update record
        rec1.compute_reproductive_value()
        # Test vs expectation
        f = np.fliplr(np.cumsum(np.fliplr(rec1["fitness_term"]),1))
        jf = np.fliplr(np.cumsum(np.fliplr(rec1["junk_fitness_term"]),1))
        assert np.allclose(rec1["repr_value"], f/rec1["cmv_surv"])
        assert np.allclose(rec1["junk_repr_value"], jf/rec1["junk_cmv_surv"])

    def test_compute_bits(self, rec1):
        """Test computation of mean and variance in bit value along
        chromosome for a genome filled with 1's."""
        rec1.compute_bits()
        # Define parameters
        m,c = rec1["number_of_snapshots"],rec1["chr_len"]
        # Test against expectation
        assert np.array_equal(rec1["n1"], np.ones([m,c]))
        assert np.array_equal(rec1["n1_var"], np.zeros([m,c]))

    def test_compute_entropies(self, rec1):
        """Test computation of per-bit and per-locus entropy in a 
        population, for a genome filled with 1's."""
        rec1.compute_entropies()
        # Define parameters
        z = np.zeros(rec1["number_of_snapshots"])
        # Test against expectation
        assert np.array_equal(rec1["entropy_bits"], z)
        assert sorted(rec1["entropy_gt"].keys()) == ["a", "n", "r", "s"]
        for v in rec1["entropy_gt"].values(): assert np.array_equal(v,z)

    def test_compute_actual_death(self, rec1):
        """Test if compute_actual_death stores expected results for
        artificial data."""
        rec1.compute_actual_death()
        r = copy.deepcopy(rec1)
        maxls = r["max_ls"][0]
        r["age_distribution"] = np.tile(1/float(maxls), (3, maxls))
        r["population_size"] = np.array([maxls*4,maxls*2,maxls])
        r.compute_actual_death()
        print r["age_distribution"].shape, r["population_size"].shape
        print r["actual_death_rate"].shape, r["number_of_stages"], r["max_ls"]
        assert np.array_equal(r["actual_death_rate"],
                np.tile(0.5, np.array(r["age_distribution"].shape) - 1))

    def test_get_window(self, rec1):
        """Test window generation on static data with random window size."""
        # Initialise
        # Generate windows for 1D and 2D data
        exp = {"population_size":rec1["population_size"][0], "n1":1}
        def test_window(key, wsize, shape, test1=True):
            w = rec1.get_window(key, wsize)
            assert w.shape == shape
            if test1: assert np.all(w == exp[key])
            else: assert np.sum(w) == 0
            return w
        for s in ["population_size", "n1"]:
            x = rec1[s]
            dim = len(x.shape)-1
            # Window right size
            ws = random.randint(1,x.shape[dim])
            w_shape = x.shape[:dim] + (x.shape[dim] - ws + 1, ws)
            w = test_window(s, ws, w_shape)
            # Window too big - should correct to dimension size
            w = test_window(s, 1e100, x.shape[:dim] + (0,x.shape[dim]+1), False)
            # Zero window
            w = test_window(s, 0, x.shape[:dim] + (x.shape[dim]+1,0), False)
            # Negative window
            with pytest.raises(ValueError):
                w = rec1.get_window(s,-1)

    def test_compute_windows(self, rec1):
        """Test generation of particular sliding window record entries on a
        degenerate population."""
        # Initialise
        rec1.compute_windows()
        # Test window entries
        exp_val = {"population_size":rec1["population_size"][0],
                "resources":rec1["resources"][0],"n1":1}
        for s in ["population_size","resources","n1"]:
            x,ws = rec1[s],rec1["windows"][s]
            dim = len(x.shape)-1
            sh = (x.shape[dim]-ws+1,) if x.shape[dim]>ws+1 else (0,)
            shape = x.shape[:dim] + sh
            assert np.array_equal(rec1[s+"_window_var"], np.zeros(shape))
            assert np.array_equal(rec1[s+"_window_mean"], 
                    np.tile(exp_val[s],shape))

    def test_finalise(self, rec1, rec2):
        """Test that finalise is equivalent to calling all finalisation
        methods separately."""
        # First check that rec1 is finalised and rec2 is not
        assert rec2["actual_death_rate"] == 0
        assert type(rec1["actual_death_rate"]) is np.ndarray
        # Then finalise rec2 and compare
        rec2.finalise()
        assert type(rec2["actual_death_rate"]) is np.ndarray
        for k in rec2.keys():
            print k
            if k in ["snapshot_pops", "final_pop"]: continue
            o1, o2 = rec1[k], rec2[k]
            if k == "actual_death_rate":
                assert o1.shape == o2.shape
                assert np.sometrue(np.isnan(o1))
                assert np.sometrue(np.isnan(o2))
                assert np.isclose(np.mean(np.isnan(o1)),np.mean(np.isnan(o2)))
            elif isinstance(o1, dict):
                for l in o1.keys():
                    assert np.array_equal(np.array(o1[l]),np.array(o2[l]))
            elif not callable(o1):
                assert np.array_equal(np.array(o1), np.array(o2))
        # TODO: Test different snapshot_pops saving options
