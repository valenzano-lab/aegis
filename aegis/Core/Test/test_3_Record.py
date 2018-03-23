from aegis.Core import Config, Population, Record
from aegis.Core import chance, init_ages, init_genomes, init_generations
from aegis.Core import init_gentimes
from aegis.Core import fivenum, make_windows
from aegis.Core.Config import deep_eq, deep_key
import pytest,importlib,types,random,copy,string
import numpy as np

#########################
## AUXILIARY FUNCTIONS ##
#########################

def static_fill(rec_obj, pop_obj):
    """Fill a whole record object with the same initial
    population state."""
    r = copy.deepcopy(rec_obj)
    n = r["n_stages"] if not r["auto"] else r["max_stages"]
    s = r["snapshot_generations"] if r["auto"] else r["snapshot_stages"]
    c = 0
    for x in xrange(n):
        snapshot = c if x in s else -1
        r.update(pop_obj, 100, 1, 1, x, snapshot)
        c += 1 if x in s else 0
    # in case not all snapshot_generations reached
    for i in xrange(r["n_snapshots"]):
        r["snapshot_pops"][i] = pop_obj.clone()
    return r

##############
## FIXTURES ##
##############

from test_1_Config import conf, conf_path, ran_str, gen_trseed

@pytest.fixture(scope="module")
def pop(request, conf):
    """Create a sample population from the default configuration."""
    gm = conf["genmap"]
    np.random.shuffle(gm)
    return Population(conf["params"], gm, init_ages(), init_genomes(),
            init_generations(), init_gentimes())

@pytest.fixture(scope="module")
def rec(request, conf):
    """Create a sample record from the default configuration."""
    return Record(conf)

@pytest.fixture(scope="module")
def pop1(request, pop):
    """Population of young adults with genomes filled with ones."""
    p = pop.clone()
    p.genomes = np.ones(p.genomes.shape).astype(int)
    p.ages = np.tile(p.maturity, p.N)
    p.generations = np.zeros(p.N, dtype=int)
    p.gentimes = np.zeros(p.N, dtype=int)
    return p

@pytest.fixture(scope="module")
def pop2(request, pop):
    """Population of young adults with general genomes."""
    p = pop.clone()
    p.ages = np.tile(p.maturity, p.N)
    p.generations = np.zeros(p.N, dtype=int)
    p.gentimes = np.zeros(p.N, dtype=int)
    return p

@pytest.fixture(scope="module")
def rec1(request, rec, pop1):
    """Record filled with initial state of pop1."""
    return static_fill(rec, pop1)

@pytest.fixture(scope="module")
def rec1_copy(request, rec, pop1):
    """Record filled with initial state of pop1."""
    return static_fill(rec, pop1)

@pytest.fixture(scope="module")
def rec2(request, rec, pop2):
    """Record filled with initial state of pop2."""
    return static_fill(rec, pop2)

###########
## TESTS ##
###########

class TestRecord:
    """Test Record object initialisation and methods."""

    ## FUNDAMENTALS ##

    def test_record_copy(self, rec):
        """Test that the config copy() method deepcopies everything but prng."""
        r1 = rec.copy()
        r2 = copy.deepcopy(rec)
        r2["prng"] = r1["prng"]
        r2["params"]["prng"] = r1["params"]["prng"]
        assert r1 == r2

    ## INITIALISATION ##

    def test_record_init(self, conf, rec):
        R = rec.copy()
        # Conf param entries should be inherited in Record
        for k in conf.keys():
            #print k, R[k], np.array(conf[k])
            if k in ["res_function", "stv_function"]: # Can't save function objects
                assert R[k] == 0
            else:
                assert deep_key(k, R, conf, True)
        # Check basic run info
        assert np.array_equal(R["dieoff"], np.array(False))
        assert not R["age_dist_truncated"]
        assert np.array_equal(R["prev_failed"], np.array(0))
        # Per-stage data entry
        n = R["n_stages"] if not R["auto"] else R["max_stages"]
        a0, a1 = np.zeros(n), np.zeros([n, R["max_ls"]])
        assert np.array_equal(R["population_size"], a0)
        assert np.array_equal(R["resources"], a0)
        if R["surv_pen"]: assert np.array_equal(R["surv_penf"], a0)
        if R["repr_pen"]: assert np.array_equal(R["repr_penf"], a0)
        assert np.array_equal(R["age_distribution"], a1)
        assert np.array_equal(R["generation_dist"], np.zeros([n,5]))
        assert np.array_equal(R["gentime_dist"], np.zeros([n,5]))
        if R["auto"]:
            assert R["age_dist_stages"] == [[] for i in xrange(R["n_snapshots"])]
        # Snapshot population placeholders
        assert R["snapshot_pops"] == [0]*R["n_snapshots"]
        # Empty names for final computation
        def iszero(*names):
            for name in names: assert R[name] == 0
        # Genotype sum statistics (density and average)
        #iszero("density_per_locus", "density", "mean_gt", "var_gt", "entropy_gt")
        # Survival and reproduction
        #iszero("cmv_surv", "junk_cmv_surv", "prob_mean", "prob_var", "junk_mean",
        #        "junk_var", "fitness_term", "junk_fitness_term", "repr_value",
        #        "junk_repr_value")
        # Per-bit statistics, actual death
        #iszero("n1", "n1_var", "entropy_bits", "actual_death_rate")
        # Sliding windows
        #iszero("population_size_window_mean", "population_size_window_var",
        #        "resources_window_mean", "resources_window_var",
        #        "n1_window_mean", "n1_window_var")

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
        rec2 = rec.copy()
        # record age_dist
        rec2.update(pop, 100, 1, 1, 0, -1, 0)
        agedist=np.bincount(pop.ages,minlength=pop.max_ls)/float(pop.N)
        assert rec2["resources"][0] == 100
        assert rec2["population_size"][0] == pop.N
        if rec2["surv_pen"]: assert rec2["surv_penf"][0] == 1
        if rec2["repr_pen"]: assert rec2["repr_penf"][0] == 1
        assert np.array_equal(rec2["age_distribution"][0], agedist)
        assert np.allclose(rec2["generation_dist"][0],
                fivenum(pop.generations))
        assert np.allclose(rec2["gentime_dist"][0],
                fivenum(pop.gentimes))
        for n in xrange(len(rec2["snapshot_pops"])):
            assert rec2["snapshot_pops"][n] == 0

    def test_update_full(self, rec, pop):
        """Test that snapshot update function records correctly."""
        rec2 = rec.copy()
        pop2 = pop.clone()
        np.random.shuffle(pop2.genmap)
        np.random.shuffle(pop2.ages)
        np.random.shuffle(pop2.genomes)
        # do not record age_dist
        rec2.update(pop2, 200, 2, 2, 0, 0, -1)
        # Per-stage factors
        assert rec2["population_size"][0] == pop2.N
        assert rec2["resources"][0] == 200
        if rec2["surv_pen"]: assert rec2["surv_penf"][0] == 2
        if rec2["repr_pen"]: assert rec2["repr_penf"][0] == 2
        assert np.array_equal(rec2["age_distribution"][0], np.zeros(rec2["max_ls"]))
        assert np.allclose(rec2["generation_dist"][0],
                fivenum(pop.generations))
        assert np.allclose(rec2["gentime_dist"][0],
                fivenum(pop.gentimes))
        for n in xrange(1,len(rec2["snapshot_pops"])):
            assert rec2["snapshot_pops"][n] == 0
        # Snapshot population
        p = rec2["snapshot_pops"][0]
        assert isinstance(p, Population)
        assert np.array_equal(p.genmap, pop2.genmap)
        assert np.array_equal(p.ages, pop2.ages)
        assert np.array_equal(p.genomes, pop2.genomes)
        assert np.array_equal(p.generations, pop2.generations)
        assert np.array_equal(p.gentimes, pop2.gentimes)


    ## FINALISATION ##

    # not computed for rec2
    def test_compute_snapshot_properties(self, rec1):
        """Test that compute_snapshot_properties performs correctly for
        a genome filled with 1's.""" #! TODO: How about in a normal case?
        n = rec1["n_stages"] if not rec1["auto"] else rec1["max_stages"]
        mt = float(rec1["maturity"])
        g = np.ceil(n/mt).astype(int)+1
        print n, mt, g
        print rec1["n_snapshots"]
        print "snapshot_pops:\n", rec1["snapshot_pops"]
        rec1.compute_snapshot_properties()
        print len(rec1["snapshot_generation_distribution"])
        # Compute expected values
        exp_dist = {"age": np.zeros(rec1["max_ls"]),
                "gentime": np.zeros(rec1["max_ls"]),
                "generation": np.zeros(2)}
        exp_dist["age"][int(mt)] = 1 # Everyone is the same age
        exp_dist["gentime"][0] = 1 # Everyone is from initial pop
        exp_dist["generation"][1] = 1 # Everyone is of generation 0
        sum_dim = {"age": 1, "gentime": 1, "generation": 2}
        # Test output
        for k in ["age", "gentime", "generation"]:
            o = rec1["snapshot_{}_distribution".format(k)]
            print exp_dist[k]
            print k,o
            assert np.all(o == exp_dist[k])
            assert np.allclose(np.sum(o, sum_dim[k]), 1)

    # DONE
    def test_compute_locus_density(self, rec1, pop2, rec2):
        """Test that compute_locus_density performs correctly for a
        genome filled with 1's and one randomly generated."""
        llist = ["a","n","r","s"]
        rec1.compute_locus_density()
        obj1 = rec1["density_per_locus"]
        rec2.compute_locus_density()
        obj2 = rec2["density_per_locus"]
        assert sorted(obj1.keys()) == llist
        assert sorted(obj2.keys()) == llist

        n_snap,l1,ns1 = rec1["n_snapshots"], rec1["max_ls"],\
                rec1["n_states"]
        g,nn,mt1 = len(rec1["genmap"]), rec1["n_neutral"], rec1["maturity"]
        dims = {"a":[n_snap,g,ns1],"n":[n_snap,nn,ns1],"r":[n_snap,l1-mt1,ns1], \
                "s":[n_snap,l1,ns1]}

        l2,mt2,ns2 = pop2.max_ls, pop2.maturity, rec2["n_states"]
        loci_all = pop2.sorted_loci()
        loci = {"s":np.array(loci_all[:,:l2]),
                "r":np.array(loci_all[:,l2:(2*l2-mt2)]),
                "n":np.array(loci_all[:,(2*l2-mt2):]), "a":loci_all}
        def density(x):
            bins = np.bincount(x,minlength=ns2)
            return bins/float(sum(bins))

        for l in llist:
            check1 = np.zeros(dims[l])
            check1[:,:,-1] = 1
            check2 = np.apply_along_axis(density,0,loci[l])
            check2 = check2.T
            assert np.array_equal(obj1[l], check1)
            assert np.allclose(obj2[l][0], check2)

    # DONE
    def test_compute_total_density(self, rec1, pop2, rec2):
        """Test that compute_total_density performs correctly for a
        genome filled with 1's and one randomly generated."""
        llist = ["a","n","r","s"]
        rec1.compute_total_density()
        obj1 = rec1["density"]
        rec2.compute_total_density()
        obj2 = rec2["density"]
        assert sorted(obj1.keys()) == llist
        assert sorted(obj2.keys()) == llist

        n_snap,ns1 = rec1["n_snapshots"], rec1["n_states"]
        l2,mt2,ns2 = pop2.max_ls, pop2.maturity, rec2["n_states"]
        loci_all = pop2.sorted_loci()
        loci = {"s":np.array(loci_all[:,:l2]),
                "r":np.array(loci_all[:,l2:(2*l2-mt2)]),
                "n":np.array(loci_all[:,(2*l2-mt2):]), "a":loci_all}
        loci_flat = copy.deepcopy(loci)
        loci_flat.update((k,v.reshape(v.shape[0]*v.shape[1])) \
                for k,v in loci_flat.items())
        loci_flat.update((k,np.bincount(v,minlength=ns2)/float(len(v))) \
                for k,v in loci_flat.items())

        for l in llist:
            check1 = np.zeros([n_snap,ns1])
            check1[:,-1] = 1
            assert np.array_equal(obj1[l], check1)
            assert np.allclose(obj2[l][0], loci_flat[l])

    # DONE
    def test_compute_genotype_mean_var(self, rec1, pop2, rec2):
        """Test that compute_genotype_mean_var performs correctly for a
        genome filled with 1's and one randomly generated."""
        rec1.compute_genotype_mean_var()
        obj1_mean = rec1["mean_gt"]
        obj1_var = rec1["var_gt"]
        rec2.compute_genotype_mean_var()
        obj2_mean = rec2["mean_gt"]
        obj2_var = rec2["var_gt"]
        llist = ["a","n","r","s"]
        assert sorted(obj1_mean.keys()) == llist
        assert sorted(obj1_var.keys()) == llist
        assert sorted(obj2_mean.keys()) == llist
        assert sorted(obj2_var.keys()) == llist

        n_snap,l1,ns1 = rec1["n_snapshots"], rec1["max_ls"],\
                rec1["n_states"]
        g,nn,mt1 = len(rec1["genmap"]), rec1["n_neutral"], rec1["maturity"]
        dims = {"a":[n_snap,g],"n":[n_snap,nn],"r":[n_snap,l1-mt1],"s":[n_snap,l1]}

        l2,mt2,ns2 = pop2.max_ls, pop2.maturity, rec2["n_states"]
        loci_all = pop2.sorted_loci()
        loci = {"s":np.array(loci_all[:,:l2]),
                "r":np.array(loci_all[:,l2:(2*l2-mt2)]),
                "n":np.array(loci_all[:,(2*l2-mt2):]), "a":loci_all}

        for l in llist:
            check1_var = np.zeros(dims[l])
            check1_mean = copy.deepcopy(check1_var)
            check1_mean[:] = ns1-1 # all genotypes maximal
            check2_mean = np.mean(loci[l],0)
            check2_var = np.var(loci[l],0)
            print obj1_var[l].shape
            print check1_var.shape
            assert np.array_equal(obj1_var[l], check1_var)
            assert np.array_equal(obj1_mean[l], check1_mean)
            assert np.allclose(obj2_mean[l][0], check2_mean)
            assert np.allclose(obj2_var[l][0], check2_var)

    def test_compute_surv_repr_probabilities_true(self, rec1, pop2, rec2):
        """Test that compute_surv_repr_probabilities_true performs
        correctly for a genome filled with 1's and one randomly generated."""
        rec1.compute_surv_repr_probabilities_true()
        rec2.compute_surv_repr_probabilities_true()
        llist = ["repr", "surv"]
        for k in ["mean", "var"]: assert sorted(rec1["prob_"+k].keys()) == llist
        for k in ["mean", "var"]: assert sorted(rec2["prob_"+k].keys()) == llist

        # Define parameters
        n_snap,l1 = rec1["n_snapshots"], rec1["max_ls"]
        ns1,mt1= rec1["n_states"], rec1["maturity"]
        dims = {"surv":[n_snap,l1], "repr":[n_snap,l1-mt1]}
        vmax = {"surv":rec1.p_surv(ns1-1), "repr":rec1.p_repr(ns1-1)}

        l2,mt2,ns2 = pop2.max_ls, pop2.maturity, rec2["n_states"]
        loci_all = pop2.sorted_loci()
        loci = {"s":np.array(loci_all[:,:l2]),
                "r":np.array(loci_all[:,l2:(2*l2-mt2)]),
                "n":np.array(loci_all[:,(2*l2-mt2):]), "a":loci_all}

        # Test vs expectation
        for l in llist:
            print l
            data = loci[l[0]]
            print "\ndata\n", data[:5]
            values = rec2[l[0]+"_range"]
            print "\nvalues\n", values
            print "\nvalues[data]\n", (values[data])[:5]
            print "...\n"
            print "\nnp.var(data[values],0)\n", np.var((values[data]),0)
            print "\nrec2\n", rec2["prob_var"][l][0]
            check_mean = np.mean(values[data],0)
            if l=="repr": check_var = np.var(values[data],0)#/20*pop2.N
            else: check_var = np.var(values[data],0)#*pop2.N
            print check_var[0]/rec2["prob_var"][l][0][0]
            print check_var[1]/rec2["prob_var"][l][0][1]
            assert np.array_equal(rec1["prob_mean"][l], np.tile(vmax[l], dims[l]))
            assert np.array_equal(rec1["prob_var"][l], np.zeros(dims[l]))
            assert np.allclose(check_mean, rec2["prob_mean"][l][0])
            assert np.allclose(check_var, rec2["prob_var"][l][0])

    # TODO general case
    def test_surv_repr_probabilities_junk(self, rec1):
        """Test that compute_surv_repr_probabilities_junk performs
        correctly for a genome filled with 1's."""
        rec1.compute_surv_repr_probabilities_junk()
        # Define parameters
        m,nn,ns = rec1["n_snapshots"],rec1["n_neutral"],rec1["n_states"]
        llist = ["repr", "surv"]
        vmax = {"surv":rec1.p_surv(ns-1), "repr":rec1.p_repr(ns-1)}
        # Test vs expectation
        for k in ["mean", "var"]: assert sorted(rec1["junk_"+k].keys()) == llist
        for l in llist:
            assert np.array_equal(rec1["junk_mean"][l], np.tile(vmax[l], [m,nn]))
            assert np.array_equal(rec1["junk_var"][l], np.zeros([m,nn]))

    # DONE
    def test_compute_cmv_surv(self, rec1, pop2, rec2):
        """Test that cumulative survival probabilities are computed
        correctly for a genome filled with 1's and one randomly generated."""
        rec1.compute_cmv_surv()
        rec2.compute_surv_repr_probabilities_junk()
        rec2.compute_cmv_surv()

        # Define parameters
        n_snap,l1,ns1 = rec1["n_snapshots"], rec1["max_ls"],\
                rec1["n_states"]
        check1 = np.tile(rec1.p_surv(ns1-1)**np.arange(l1), [n_snap,1])

        l2,mt2,ns2 = pop2.max_ls, pop2.maturity, rec2["n_states"]
        loci_all = pop2.sorted_loci()
        loci = {"s":np.array(loci_all[:,:l2]),
                "r":np.array(loci_all[:,l2:(2*l2-mt2)]),
                "n":np.array(loci_all[:,(2*l2-mt2):]), "a":loci_all}
        values = rec2["s_range"]
        check2 = np.ones(l2)
        check2[1:] = np.cumprod(np.mean(values[loci["s"]],0))[:-1]
        check2_junk = np.ones(l2)
        check2_junk[1:] = np.cumprod(np.tile(np.mean(values[loci["n"]]),l2-1))
        # Test vs expectation
        assert np.allclose(rec1["cmv_surv"], check1)
        assert np.allclose(rec1["junk_cmv_surv"], check1)
        assert np.allclose(rec2["cmv_surv"][0], check2)
        assert np.allclose(rec2["junk_cmv_surv"][0], check2_junk)

    # DONE
    def test_compute_mean_repr(self, rec1, pop2, rec2):
        """Test that mean reproduction probability calculations are
        computed correctly for a genome filled with 1's and one randomly generated.
        """
        rec1.compute_mean_repr()
        rec2.compute_mean_repr()

        # Define parameters
        sex = rec1["repr_mode"] in ["sexual", "assort_only"]
        div = 2.0 if sex else 1.0
        n_snap,ns1 = rec1["n_snapshots"], rec1["n_states"]
        l1,mt1 = rec1["max_ls"], rec1["maturity"]
        mr = np.tile(rec1.p_repr(ns1-1), [n_snap,l1])
        mr[:,:mt1] = 0

        l2,mt2,ns2 = pop2.max_ls, pop2.maturity, rec2["n_states"]
        loci_all = pop2.sorted_loci()
        loci = {"s":np.array(loci_all[:,:l2]),
                "r":np.array(loci_all[:,l2:(2*l2-mt2)]),
                "n":np.array(loci_all[:,(2*l2-mt2):]), "a":loci_all}
        values = rec2["r_range"]
        check2 = np.zeros(l2)
        check2[mt2:] = np.mean(values[loci["r"]],0)[:]
        check2_junk = np.zeros(l2)
        check2_junk[mt2:] = np.tile(np.mean(values[loci["n"]]), l2-mt2)[:]

        # Test vs expectation
        assert np.allclose(rec1["mean_repr"], mr/div)
        assert np.allclose(rec1["junk_repr"], mr/div)
        assert np.allclose(rec2["mean_repr"], check2/div)
        assert np.allclose(rec2["junk_repr"][0], check2_junk/div)

    # DONE
    def test_compute_fitness(self, rec1, rec2):
        """Test that per-age and total fitness are computed correctly
        for a genome filled with 1's and one randomly generated."""
        # Update record
        rec1.compute_fitness()
        rec2.compute_fitness()
        # 1's genome
        assert np.allclose(rec1["fitness_term"],
                rec1["cmv_surv"]*rec1["mean_repr"])
        assert np.allclose(rec1["junk_fitness_term"],
                rec1["junk_cmv_surv"]*rec1["junk_repr"])
        assert np.allclose(rec1["fitness"], np.sum(rec1["fitness_term"], 1))
        assert np.allclose(rec1["junk_fitness"],
                np.sum(rec1["junk_fitness_term"], 1))
        # random genome
        assert np.allclose(rec2["fitness_term"],
                rec2["cmv_surv"]*rec2["mean_repr"])
        assert np.allclose(rec2["junk_fitness_term"],
                rec2["junk_cmv_surv"]*rec2["junk_repr"])
        assert np.allclose(rec2["fitness"], np.sum(rec2["fitness_term"], 1))
        assert np.allclose(rec2["junk_fitness"],
                np.sum(rec2["junk_fitness_term"], 1))

    # DONE
    def test_compute_reproductive_value(self, rec1, rec2):
        # Update record
        rec1.compute_reproductive_value()
        rec2.compute_reproductive_value()
        # Test vs expectation
        f1 = np.fliplr(np.cumsum(np.fliplr(rec1["fitness_term"]),1))
        jf1 = np.fliplr(np.cumsum(np.fliplr(rec1["junk_fitness_term"]),1))
        f2 = np.fliplr(np.cumsum(np.fliplr(rec2["fitness_term"]),1))
        jf2 = np.fliplr(np.cumsum(np.fliplr(rec2["junk_fitness_term"]),1))
        assert np.allclose(rec1["repr_value"], f1/rec1["cmv_surv"])
        assert np.allclose(rec1["junk_repr_value"], jf1/rec1["junk_cmv_surv"])
        assert np.allclose(rec2["repr_value"], f2/rec2["cmv_surv"])
        assert np.allclose(rec2["junk_repr_value"], jf2/rec2["junk_cmv_surv"])

    # DONE
    def test_compute_bits(self, rec1, pop2, rec2):
        """Test computation of mean and variance in bit value along
        chromosome for a genome filled with 1's and one randomly generated."""
        rec1.compute_bits()
        rec2.compute_bits()

        # 1's genome
        n_snap,chr_len1 = rec1["n_snapshots"],rec1["chr_len"]
        assert np.array_equal(rec1["n1"], np.ones([n_snap,chr_len1]))
        assert np.array_equal(rec1["n1_var"], np.zeros([n_snap,chr_len1]))

        # random genome
        nbase = rec2["n_base"]
        # choose one locus at random and fill with ones to check for correct order
        rbit = np.random.choice(xrange(len(pop2.genmap)))
        print "rbit: ", rbit
        print "pop2.genmap[rbit]: ", pop2.genmap[rbit]
        rlocus = np.array([rbit*nbase + c for c in xrange(nbase)])
        rlocus = np.stack([rlocus, rlocus+pop2.chr_len])
        # get the regarding locus in sorted genome
        apo = pop2.genmap[rbit] # age + offset
        if apo/pop2.neut_offset > 0:
            abit = 2*pop2.max_ls - pop2.maturity + apo%pop2.neut_offset
        elif apo/pop2.repr_offset > 0:
            abit = pop2.max_ls - pop2.maturity + apo%pop2.repr_offset
        else:
            abit = apo
        alocus = np.array([abit*nbase + c for c in xrange(nbase)])
        genomes = copy.deepcopy(pop2.genomes)
        genomes[:,rlocus] = 1
        genomes = genomes.reshape(pop2.N*2, pop2.chr_len)
        order = np.ndarray.flatten(np.array([pop2.genmap_argsort*nbase + c for c in\
                xrange(nbase)]), order="F")
        check1 = np.mean(genomes,0)[order]
        check11 = rec2["n1"][0]
        check11[alocus] = 1
        check2 = np.var(genomes,0)[order]
        check22 = rec2["n1_var"][0]
        check22[alocus] = 0
        np.set_printoptions(threshold=np.nan)
        assert np.allclose(check11, check1)
        assert np.allclose(check22, check2)

    def test_reorder_bits(self, rec1):
        R = rec1.copy()
        genmap = copy.deepcopy(R["genmap"])
        rofs = R["repr_offset"]
        nofs = R["neut_offset"]
        maxls = R["max_ls"]
        m = R["maturity"]
        nsnap = 5
        nb = R["n_base"]
        # generate n1
        R["n_snapshots"] = nsnap
        R["n1"] = np.random.random((nsnap,R["chr_len"]))
        # general
        ixr = np.logical_and(genmap>rofs,genmap<nofs)
        genmap[ixr] = genmap[ixr]-rofs-m+maxls
        ixn = genmap>=nofs
        genmap[ixn] = genmap[ixn]-nofs+2*maxls-m
        n1s = copy.deepcopy(R["n1"])
        n1s = n1s.reshape((nsnap,n1s.shape[1]/nb,nb))
        n1s = n1s[:,genmap]
        n1s = n1s.reshape((nsnap,n1s.shape[1]*nb))
        R.reorder_bits()
        assert np.array_equal(genmap,R["genmap_ix"])
        assert np.array_equal(R["n1_reorder"],n1s)
        # check that for ordered genmap it does nothing
        R["genmap"].sort()
        exp = copy.deepcopy(R["n1"])
        R.reorder_bits()
        assert np.array_equal(R["n1_reorder"],exp)

    # TODO general case
    def test_compute_entropies(self, rec1):
        """Test computation of per-bit and per-locus entropy in a
        population, for a genome filled with 1's."""
        rec1.compute_entropies()
        # Define parameters
        z = np.zeros(rec1["n_snapshots"])
        # Test against expectation
        assert np.array_equal(rec1["entropy_bits"], z)
        assert sorted(rec1["entropy_gt"].keys()) == ["a", "n", "r", "s"]
        for v in rec1["entropy_gt"].values(): assert np.array_equal(v,z)

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
            w = test_window(s, int(1e100), x.shape[:dim] + (0,x.shape[dim]+1), False)
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

    def test_finalise(self, rec1, rec1_copy):
        """Test that finalise is equivalent to calling all finalisation
        methods separately."""
        # First check that rec1 is finalised and rec1_copy is not
        assert "n1_window_var" in rec1.keys()
        assert not "n1_window_var" in rec1_copy.keys()
        # Account dor methods in finalisation that were not called in the above
        # tests: truncate age distribution
        rec1["age_dist_truncated"] = True
        rec1_copy["age_dist_truncated"] = True
        # Then finalise rec1_copy and compare
        rec1["finalised"] = True
        rec1_copy.finalise()
        # Check post finalisation methods not in finalisation
        assert not "actual_death_rate" in rec1_copy.keys()
        assert not "snapshot_age_distribution_avrg" in rec1_copy.keys()
        # Compare finalised entries
        for k in rec1_copy.keys():
            print k
            if k in ["snapshot_pops","final_pop","snapshot_age_distribution"]:
                continue
            o1, o2 = rec1[k], rec1_copy[k]
            if isinstance(o1, dict):
                for l in o1.keys():
                    if l=="prng":
                        assert np.array_equal(o1[l].rand(100),o2[l].rand(100))
                    else:
                        assert np.array_equal(np.array(o1[l]),np.array(o2[l]))
            elif isinstance(o1, np.random.RandomState):
                assert np.array_equal(o1.rand(100),o2.rand(100))
            elif isinstance(o1, tuple): # numpy pseudo-random number generator seed
                for w1,w2 in zip(o1,o2):
                    if isinstance(w1, np.ndarray): assert np.array_equal(w1,w2)
                    else: return w1==w2
            elif not callable(o1):
                assert np.array_equal(np.array(o1), np.array(o2))
        # TODO: Test different snapshot_pops saving options

    ## FINALISATION NEEDING SPECIAL SETUP ##

    def test_truncate_age_dist_stages(self, rec):
        if not rec["auto"]: return
        R = rec.copy()
        # truncates to snapshots taken
        R.truncate_age_dist_stages()
        assert R["age_dist_stages"].shape[0] == R["n_snapshots"]
        # empty
        l = [[] for i in xrange(R["n_snapshots"])]
        R["age_dist_stages"] = copy.copy(l)
        R.truncate_age_dist_stages()
        assert np.array_equal(R["age_dist_stages"], np.array(l))
        # only one sublist
        R["n_snapshots"] = 1
        R["age_dist_stages"] = copy.copy(l)
        R["age_dist_stages"][0] = range(10)
        R.truncate_age_dist_stages()
        assert np.array_equal(R["age_dist_stages"], np.array([range(10)]))
        # general case
        m_snap = np.random.randint(2,20)
        ix = np.random.randint(0,np.random.randint(1000,10000),m_snap)
        ix.sort()
        ads = [range(ix[2*i],ix[2*i+1]) for i in range(ix.shape[0]/2)]
        minl = ix[1] - ix[0]
        for i in range(1,m_snap/2):
            minl = min(minl,ix[2*i+1] - ix[2*i])
        R["age_dist_stages"] = copy.deepcopy(ads)
        R["n_snapshots"] = m_snap/2
        R.truncate_age_dist_stages()
        # prints
        print "auto: ", R["auto"]
        print "n snap:", m_snap/2
        print "age_dist_stages before truncation:"
        print "shape: ", np.array(ads).shape
        print ads
        print "age_dist_stage after truncation:"
        print "shape: ", R["age_dist_stages"].shape
        print R["age_dist_stages"]
        if minl/2==0 or m_snap/2==1:
            assert R["age_dist_stages"].shape == (m_snap/2, minl)
        else: assert R["age_dist_stages"].shape == (m_snap/2, minl/2*2)

    def test_truncate_age_dist(self, rec, pop):
        """Test only wished age distribution entries are present after
        truncation."""
        rec2 = rec.copy()
        rec2.update(pop, 100, 1, 1, 0, -1, 0)
        if rec2["age_dist_N"] == "all":
            exp = copy.deepcopy(rec2["age_distribution"])
            rec2.truncate_age_dist()
            assert np.array_equal(rec2["age_distribution"], exp)
        else:
            if rec2["auto"]:
                # update age_dist_stages since this is usually done in Run
                rec2["age_dist_stages"][0] = [0]
                ix = np.array([0])
                print "auto"
            else:
                ix = rec2["age_dist_stages"][:1].flatten()
                print "no auto"
            print ix
            exp = copy.deepcopy(rec2["age_distribution"][ix])
            exp = exp.reshape((1,exp.shape[0],exp.shape[1]))
            rec2["n_snapshots"] = 1
            rec2.truncate_age_dist()
            print "rec2['age_distribution']"
            print rec2["age_distribution"].shape
            print "exp"
            print exp.shape
            assert np.array_equal(rec2["age_distribution"], exp)

    def test_truncate_age_dist2(self,rec):
        R = rec.copy()
        # if all, does nothing
        if R["age_dist_N"] == "all":
            exp = copy.deepcopy(R["age_distribution"])
            R.truncate_age_dist()
            assert R["age_dist_truncated"]
            assert np.array_equal(R["age_distribution"],exp)
            return
        # if already truncated, does nothing
        R["age_dist_truncated"] = True
        R["age_dist_stages"] = range(10)
        R.truncate_age_dist()
        assert R["age_dist_stages"] == range(10)
        # general
        R["age_dist_truncated"] = False
        maxls = R["max_ls"]
        nsnap = R["n_snapshots"]
        nstage = np.random.randint(10000)
        means = np.linspace(0,nstage,nsnap).astype(int)
        minl = np.random.randint(1,(means[1]-means[0])/2)
        R["age_dist_stages"] = make_windows(means,minl)
        R["age_distribution"] = np.random.random((nstage,maxls))

        ix = np.array(R["age_dist_stages"]).flatten()
        exp = copy.deepcopy(R["age_distribution"][ix])
        exp = exp.reshape((nsnap, exp.size/nsnap/maxls, maxls))

        R.truncate_age_dist(False)
        assert R["age_distribution"].shape[0] == nsnap
        assert R["age_distribution"].shape[1] == exp.size/nsnap/maxls
        assert R["age_distribution"].shape[2] == maxls
        assert np.array_equal(R["age_distribution"],exp)

    ## POST FINALISATION (PLOTTER) ##

    def test_compute_actual_death(self, rec1):
        """Test if compute_actual_death stores expected results for
        artificial data."""
        r = rec1.copy()
        maxls = r["max_ls"]
        if r["age_dist_N"] == "all":
            r["age_distribution"] = np.tile(1/float(maxls), (3, maxls))
            r["population_size"] = np.array([maxls*4,maxls*2,maxls])
        else:
            r["n_snapshots"] = n_snap = 2
            r["age_distribution"] = np.tile(1/float(maxls), (n_snap,10,maxls))
            r["age_dist_truncated"] = True
            r["population_size"] = np.tile(maxls*(2**np.arange(9,-1,-1)),10)
            r["age_dist_stages"] = np.vstack((np.arange(10),np.arange(10)+90))
        r.compute_actual_death()
        n = r["n_stages"] if not r["auto"] else r["max_stages"]
        print r["age_distribution"].shape, r["population_size"].shape
        print r["actual_death_rate"].shape, n, r["max_ls"]
        if r["age_dist_N"] == "all":
            assert np.array_equal(r["actual_death_rate"],
                np.tile(0.5, np.array(r["age_distribution"].shape) - 1))
        else:
            assert np.array_equal(r["actual_death_rate"],
                np.tile(0.5, np.array([n_snap,r["age_distribution"].shape[1]\
                        -1, r["age_distribution"].shape[2]-1])))

    def test_compute_snapshot_age_dist_avrg(self, rec2):
        """Test snapshot age distribution averaging for randomly generated
        and already truncated ge distribution entries."""
        r = rec2.copy()
        # If all recorded, calling the method should do nothing
        if r["age_dist_N"] == "all":
            r.compute_snapshot_age_dist_avrg()
            assert not "snapshot_age_distribution_avrg" in r.keys()
            return
        # Otherwise setup record and do the check
        n_snap = r["n_snapshots"]
        maxls = r["max_ls"]
        ns = np.random.randint(50)*2/2 # number of stages per snap
        age_dist = np.random.random((n_snap*ns,maxls))
        age_dist = age_dist / np.nanmean(age_dist,1)[:,None]
        age_dist = np.reshape(age_dist,(n_snap,ns,maxls))
        r["age_distribution"] = np.copy(age_dist)
        r["age_dist_stages"] = np.arange(1) # just needs to have size > 0
        r["age_dist_truncated"] = True
        # Compute expected output
        exp = np.nanmean(age_dist,1)
        r.compute_snapshot_age_dist_avrg()
        print "record:\n", r["snapshot_age_distribution_avrg"].shape
        print r["snapshot_age_distribution_avrg"]
        print "expected:\n", exp.shape
        print exp
        assert np.array_equal(r["snapshot_age_distribution_avrg"], exp)
