import pyximport; pyximport.install()
from gs_core import Simulation, Run, Population, Record, chance
import pytest, random, string, subprocess, math, copy, os, sys
import numpy as np
import scipy.stats as st
from scipy.misc import comb

runChanceTests=True
runPopulationTests=True
runRecordTests=True
runRunTests=False
runSimulationTests=False

####################
### 0: FIXTURES  ###
####################

@pytest.fixture()
def ran_str(request):
    """Generate a random lowercase ascii string."""
    return \
        ''.join(random.choice(string.ascii_lowercase) for _ in range(50))

@pytest.fixture(params=["import", "random", "random"])
def conf(request):
    """Create a default configuration object."""
    S = Simulation("config_test", "", 10, False)
    c = S.get_conf("config_test")
    S.gen_conf(c)
    c.number_of_stages = 100
    if request.param == "random":
        # Randomise fundamental parameters
        c.g_dist_s, c.g_dist_r, c.g_dist_n  = np.random.uniform(size=3)
        db_low, rb_low = np.random.uniform(size=2)
        db_high = db_low + random.uniform(0, 1-db_low)
        rb_high = rb_low + random.uniform(0, 1-rb_low)
        c.death_bound,c.repr_bound = [db_low, db_high],[rb_low, rb_high]
        c.r_rate, c.m_rate, c.m_ratio = np.random.uniform(size=3)
        c.max_ls = random.randint(20, 99)
        c.maturity = random.randint(5, c.max_ls-1)
        #c.n_neutral = random.randint(1, 100)
        c.n_base = random.randint(5, 25)
        c.surv_pen = random.choice([True, False])
        c.repr_pen = random.choice([True, False])
        c.death_inc = random.randint(1, 10)
        c.repr_dec = random.randint(1, 10)
        gm_len = c.max_ls + (c.max_ls - c.maturity) + c.n_neutral
        c.window_size = random.randint(1, gm_len*c.n_base)
        S.gen_conf(c)
    return c

@pytest.fixture()
def spop(request, conf):
    """Create a sample population from the default configuration."""
    return Population(conf.params, conf.genmap, np.array([-1]),
            np.array([[-1],[-1]]))

@pytest.fixture()
def parents(request, conf):
    """Returns population of two sexual adults."""
    params = conf.params.copy()
    params["sexual"] = True
    params["age_random"] = False
    params["start_pop"] = 2
    return Population(params, conf.genmap, np.array([-1]),
            np.array([[-1],[-1]]))

@pytest.fixture()
def pop1(request, spop):
    """Create population of young adults with genomes filled with ones."""
    pop = spop.clone()
    pop.genomes = np.ones(pop.genomes.shape).astype(int)
    pop.ages = np.tile(pop.maturity, pop.N)
    return pop

@pytest.fixture()
def record(request,pop1,conf):
    """Create a record from pop1 as defined in configuration file."""
    spaces = 2 * pop1.nbase + 1
    return Record(pop1, conf.snapshot_stages, conf.number_of_stages, 
            np.linspace(1,0,spaces), np.linspace(0,1,spaces),
            conf.window_size)

####################
### 0: DUMMY RUN ###
####################

@pytest.mark.skip(reason="New setup needed.")
@pytest.mark.xfail
def test_sim_run_0():
    # Begin by running a dummy simulation and saving the output
    # Also functions as test of output functions
    with pytest.warns(Warning) as e_info:
        scriptdir = os.path.split(os.path.realpath(__file__))[0]
        os.chdir(scriptdir)
        subprocess.call(["python", "genome_simulation.py", "."])
        os.rename("run_1_pop.txt", "sample_pop.txt")
        os.rename("run_1_rec.txt", "sample_rec.txt")

#########################
### 1: FREE FUNCTIONS ###
#########################

@pytest.mark.skip(reason="Moved to Simulation class.")
@pytest.mark.xfail
class TestConfig:
    """Test that the initial simulation configuration is performed
    correctly."""

    def test_get_dir_good(self):
        """Verify that fn.get_dir functions correctly when given (a) the
        current directory, (b) the parent directory, (c) the root
        directory."""
        old_dir = os.getcwd()
        old_path = sys.path[:]
        fn.get_dir(old_dir)
        same_dir = os.getcwd()
        same_path = sys.path[:]
        assert same_dir == old_dir
        assert same_path == old_path
        test = (same_dir == old_dir and same_path == old_path)
        if old_dir != "/":
            fn.get_dir("..")
            par_dir = os.getcwd()
            par_path = sys.path[:]
            exp_path = [par_dir] + [x for x in old_path if x != old_dir]
            assert par_dir == os.path.split(old_dir)[0]
            assert par_path == exp_path
            if par_dir != "/":
                fn.get_dir("/")
                root_dir = os.getcwd()
                root_path = sys.path[:]
                exp_path = ["/"] + [x for x in par_path if x != par_dir]
                assert root_dir=="/"
                assert root_path==exp_path
            fn.get_dir(old_dir)

    def test_get_dir_bad(self, ran_str):
        """Verify that fn.get_dir throws an error when the target directory 
        does not exist."""
        with pytest.raises(SystemExit) as e_info: fn.get_dir(ran_str)

    def test_get_conf_good(self):
        """Test that fn.get_conf on the config template file returns a valid
        object of the expected composition."""
        c = fn.get_conf("config_test")
        def alltype(keys,typ):
            """Test whether all listed config items are of the 
            specified type."""
            return np.all([type(c.__dict__[x]) is typ for x in keys])
        assert alltype(["number_of_runs", "number_of_stages",
            "number_of_snapshots", "res_start", "R", "res_limit",
            "start_pop", "max_ls", "maturity", "n_base",
            "death_inc", "repr_dec", "window_size", "chr_len"], int)
        assert alltype(["crisis_sv", "V", "r_rate", "m_rate", "m_ratio"],
                    float)
        assert alltype(["sexual", "res_var", "age_random", "surv_pen",
                "repr_pen"], bool)
        assert alltype(["death_bound", "repr_bound", "crisis_stages"],
                    list)
        assert alltype(["g_dist", "params"], dict)
        assert alltype(["genmap", "d_range", "r_range", 
                "snapshot_stages"], np.ndarray)

    @pytest.mark.parametrize("sexvar,nsnap", [(True, 10), (False, 0.1)])
    def test_gen_conf(self, conf, sexvar, nsnap):
        """Test that gen_conf correctly generates derived simulation params."""
        c = fn.get_conf("config_test")
        c.sexual = sexvar
        crb1 = c.repr_bound[1]
        d = fn.gen_conf(conf)
        assert d.g_dist["s"] == c.g_dist_s
        assert d.g_dist["r"] == c.g_dist_r
        assert d.g_dist["n"] == c.g_dist_n
        assert len(c.genmap) == c.max_ls + (c.max_ls-c.maturity) + c.n_neutral
        assert d.chr_len == len(c.genmap) * c.n_base
        assert d.repr_bound[1]/crb1 == 2 if sexvar else 1
        assert (d.d_range == np.linspace(c.death_bound[1], c.death_bound[0],
                2*c.n_base+1)).all()
        assert (d.r_range == np.linspace(c.repr_bound[0], c.repr_bound[1],
                2*c.n_base+1)).all()
        assert len(d.snapshot_stages) == c.number_of_snapshots if \
                type(nsnap) is int else int(nsnap * c.number_of_stages)
        assert np.all(d.snapshot_stages == np.around(
            np.linspace(0, c.number_of_stages-1, d.number_of_snapshots), 0))
        assert d.params["sexual"] == sexvar
        assert d.params["chr_len"] == d.chr_len
        assert d.params["n_base"] == c.n_base
        assert d.params["maturity"] == c.maturity
        assert d.params["max_ls"] == c.max_ls
        assert d.params["age_random"] == c.age_random
        assert d.params["start_pop"] == c.start_pop
        assert d.params["g_dist"] == d.g_dist

    def test_get_conf_bad(self, ran_str):
        """Verify that fn.get_dir throws an error when the target file 
        does not exist."""
        with pytest.raises(IOError) as e_info: fn.get_conf(ran_str)

    def test_get_startpop_good(self):
        """Test that a blank seed returns a blank string and a valid seed
        returns a population array of the correct size."""
        assert fn.get_startpop("") == ""
        p = fn.get_startpop("sample_pop.txt")
        assert p.genomes.shape == (p.N, 2*p.chrlen)

    def test_get_startpop_bad(self, ran_str):
        """Verify that fn.get_startpop throws an error when the target 
        file does not exist."""
        with pytest.raises(IOError) as e_info: fn.get_startpop(ran_str)

# -------------------------
# RANDOM NUMBER GENERATION
# -------------------------
@pytest.mark.skipif(not runChanceTests, 
        reason="Not running chance tests.")
class TestChance:
    "Test that random binary array generation is working correctly."""

    @pytest.mark.parametrize("p", [0,1])
    def test_chance_degenerate(self, p):
        """Tests wether p=1 returns True/1 and p=0 returns False/0."""
        shape=(1000,1000)
        assert (chance(p, shape).astype(int) == p).all()

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_chance(self, p):
        precision = 0.01
        """Test that the shape of the output is correct and that the mean
        over many trials is close to the expected value."""
        shape=(1000,1000)
        c = chance(p, shape)
        s = c.shape
        assert c.shape == shape and c.dtype == "bool"
        assert abs(p-np.mean(c)) < precision

@pytest.mark.skip(reason="Moved to Run class.")
@pytest.mark.xfail
class TestUpdateResources:
    """Confirm that resources are updated correctly in the variable-
    resources condition."""

    def test_update_resources_bounded(self):
        """Confirm that resources cannot exceed upper bound or go below
        zero."""
        assert fn.update_resources(5000, 0, 1000, 2, 5000) == 5000
        assert fn.update_resources(0, 5000, 1000, 2, 5000) == 0
    
    def test_update_resources_unbounded(self):
        """Test resource updating between bounds."""
        assert fn.update_resources(1000, 500, 1000, 2, 5000) == 2000
        assert fn.update_resources(500, 1000, 1000, 2, 5000) == 500

######################
### 2: POPULATION  ###
######################

@pytest.mark.skipif(not runPopulationTests, 
        reason="Not running Population tests.")
class TestPopulationClass:
    """Test population object methods."""
    
    # Initialisation
    def test_init_population(self, record, conf):
        """Test that population parameters are correct for random and
        nonrandom ages."""
        precision = 1.1
        conf.params["start_pop"] = 2000
        conf.params["age_random"] = False
        pop_a = Population(conf.params, conf.genmap, np.array([-1]),
            np.array([[-1],[-1]]))
        conf.params["age_random"] = True
        pop_b = Population(conf.params, conf.genmap, np.array([-1]),
            np.array([[-1],[-1]]))
        print np.mean(pop_b.ages)
        print abs(np.mean(pop_b.ages)-pop_b.maxls/2)
        assert pop_a.sex == pop_b.sex == conf.params["sexual"]
        assert pop_a.chrlen == pop_b.chrlen == conf.params["chr_len"]
        assert pop_a.nbase == pop_b.nbase == conf.params["n_base"]
        assert pop_a.maxls == pop_b.maxls == conf.params["max_ls"]
        assert pop_a.maturity==pop_b.maturity == conf.params["maturity"]
        assert pop_a.N == pop_b.N == conf.params["start_pop"]
        assert (pop_a.index==np.arange(conf.params["start_pop"])).all()
        assert (pop_b.index==np.arange(conf.params["start_pop"])).all()
        assert (pop_a.genmap == conf.genmap).all()
        assert (pop_b.genmap == conf.genmap).all()
        assert (pop_a.ages == pop_a.maturity).all()
        assert not (pop_b.ages == pop_b.maturity).all()
        assert abs(np.mean(pop_b.ages)-pop_b.maxls/2) < precision
    
    # Genome array
    """Test that new genome arrays are generated correctly."""
    genmap_simple = np.append(np.arange(25), 
            np.append(np.arange(24)+100, 200))
    genmap_shuffled = np.copy(genmap_simple)
    random.shuffle(genmap_shuffled)

    @pytest.mark.parametrize("gd", [
        {"s":random.random(), "r":random.random(), "n":random.random()},
        {"s":0.5, "r":0.5, "n":0.5},
        {"s":0.1, "r":0.9, "n":0.4}])
    @pytest.mark.parametrize("gm", [genmap_simple, genmap_shuffled])
    def test_make_genome_array(self, spop, gm, gd):
        """Test that genome array is of the correct size and that
        the loci are distributed correctly."""
        pop = spop.clone()
        precision = 0.05
        loci = {
            "s":np.nonzero(gm<100)[0],
            "r":np.nonzero(np.logical_and(gm>=100,gm<200))[0],
            "n":np.nonzero(gm>=200)[0]
            }
        n = 1000
        pop.nbase = b = 10
        pop.chrlen = chr_len = 500
        pop.genmap = gm
        ga = pop.make_genome_array(n, gd)
        assert ga.shape == (n, 2*chr_len)
        condensed = np.mean(ga, 0)
        condensed = np.array([np.mean(condensed[x*b:(x*b+b-1)]) \
                for x in range(chr_len/5)])
        for k in loci.keys():
            pos = np.array([range(b) + x for x in loci[k]*b])
            pos = np.append(pos, pos + chr_len)
            tstat = abs(np.mean(ga[:,pos])-gd[k])
            assert tstat < precision

    # Minor methods

    def test_shuffle(self, spop):
        """Test if all ages, therefore individuals, present before the 
        shuffle are also present after it."""
        spop2 = spop.clone() # clone tested separately
        spop2.shuffle()
        is_shuffled = \
                not (spop.genomes == spop2.genomes).all()
        spop.ages.sort()
        spop2.ages.sort()
        assert is_shuffled
        assert (spop.ages == spop2.ages).all()

    def test_clone(self, spop):
        """Test if cloned population is identical to parent population, 
        by comparing params, ages, genomes."""
        spop2 = spop.clone()
        assert spop.params() == spop2.params()
        assert (spop.genmap == spop2.genmap).all()
        assert (spop.ages == spop2.ages).all()
        assert (spop.genomes == spop2.genomes).all()

    def test_increment_ages(self, spop):
        """Test if all ages are incrementd by one."""
        ages1 = np.copy(spop.ages)
        spop.increment_ages()
        ages2 = spop.ages
        assert (ages1+1 == ages2).all()

    def test_params(self, spop):
        """Test that params returns (at least) the 
        required information."""
        p = spop.params()
        assert len(p) >= 5 and p["sexual"] == spop.sex
        assert p["chr_len"] == spop.chrlen
        assert p["n_base"] == spop.nbase
        assert p["max_ls"] == spop.maxls
        assert p["maturity"] == spop.maturity

    def test_addto(self, spop):
        """Test if a population is successfully appended to the receiver
        population, which remains unchanged, by appending a population to
        itself."""
        pop_a = spop.clone()
        pop_b = spop.clone()
        pop_b.addto(pop_a)
        assert (pop_b.ages == np.tile(pop_a.ages,2)).all()
        assert (pop_b.genomes == np.tile(pop_a.genomes,(2,1))).all()

    # Death and crisis

    @pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
    @pytest.mark.parametrize("adults_only,offset",[(False,0),(True,100)])
    def test_get_subpop(self, spop, p, adults_only, offset):
        precision = 0.1
        """Test if the probability of passing is close to that indicated
        by the genome (when all loci have the same distribution)."""
        pop = spop.clone()
        loci = np.nonzero(
                np.logical_and(pop.genmap >= offset, pop.genmap < offset + 100)
                )[0]
        pos = np.array([range(pop.nbase) + y for y in loci*pop.nbase])
        pos = np.append(pos, pos + pop.chrlen)
        min_age = pop.maturity if adults_only else 0
        pop.genomes[:,pos] = chance(p, pop.genomes[:,pos].shape).astype(int)
        subpop = pop.get_subpop(min_age, pop.maxls, offset,
                np.linspace(0,1,2*pop.nbase + 1))
        assert abs(np.sum(subpop)/float(pop.N) - p)*(1-min_age/pop.maxls) < \
                precision

    @pytest.mark.parametrize("offset",[0, 100])
    def test_get_subpop_extreme_values(self, spop, offset):
        """Test that get_subpop correctly handles (i.e. ignores)
        individuals outside specified age range."""
        pop = spop.clone()
        pop.ages = np.random.choice([pop.maturity-1, pop.maxls],
                pop.N)
        subpop = pop.get_subpop(pop.maturity, pop.maxls, offset,
                np.linspace(0,1,2*pop.nbase + 1))
        assert np.sum(subpop) == 0

    @pytest.mark.parametrize("p", [0.0, 0.3, 0.8, 1.0])
    @pytest.mark.parametrize("x", [1.0, 3.0, 9.0])
    def test_death(self, spop, p, x):
        """Test if self.death() correctly inverts death probabilities
        and incorporates starvation factor to get survivor probabilities
        and survivor array."""
        precision = 0.15
        pop = spop.clone()
        b = pop.nbase
        surv_loci = np.nonzero(spop.genmap<100)[0]
        surv_pos = np.array([range(b) + y for y in surv_loci*b])
        surv_pos = np.append(surv_pos, surv_pos + pop.chrlen)
        pop.genomes[:, surv_pos] =\
                chance(p, pop.genomes[:, surv_pos].shape).astype(int)
        # (specifically modify survival loci only)
        pop2 = pop.clone()
        print pop2.genomes[:, surv_pos] 
        pop2.death(np.linspace(1,0,2*pop2.nbase+1), x)
        pmod = max(0, min(1, (1-x*(1-p))))
        assert abs(pop2.N/float(pop.N) - pmod) < precision

    def test_death_extreme_starvation(self, spop):
        """Confirm that death() handles extreme starvation factors 
        correctly (probability limits at 0 and 1)."""
        pop0 = spop.clone()
        pop1 = spop.clone()
        pop2 = spop.clone()
        drange = np.linspace(1, 0.001, 2*spop.nbase+1)
        pop0.death(drange, 1e10)
        pop1.death(drange, -1e10)
        pop2.death(drange, 1e-10)
        assert pop0.N == 0
        assert pop1.N == spop.N
        assert pop2.N == spop.N

    @pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
    def test_crisis(self, spop,p):
        """Test whether extrinsic death crisis removes the expected
        fraction of the population."""
        precision = 0.1
        pop = spop.clone()
        pop.crisis(p)
        assert abs(pop.N - p*spop.N) < precision

    # Reproduction

    def test_recombine_none(self, spop):
        """Test if genome stays same if recombination chance is zero."""
        pop = spop.clone()
        pop.recombine(0)
        assert (pop.genomes == spop.genomes).all()

    def test_recombine_all(self, conf):
        """Test if resulting genomee is equal to recombine_zig_zag when
        recombination chance is one."""
        def recombine_zig_zag(pop):
            """Recombine the genome like so:
            before: a1-a2-a3-a4-b1-b2-b3-b4
            after:  b1-a2-b3-a4-a1-b2-a3-b4."""
            g = pop.genomes.copy()
            h = np.copy(g[:,:pop.chrlen:2])
            g[:,:pop.chrlen:2] = g[:,pop.chrlen::2]
            g[:,pop.chrlen::2] = h
            return g
        conf.params["start_pop"] = 10
        pop = Population(conf.params, conf.genmap, np.array([-1]),
            np.array([[-1],[-1]]))
        zz = recombine_zig_zag(pop)
        pop.recombine(1)
        assert (pop.genomes == zz).all()

    def test_assortment(self, parents):
        """Test if assortment of two adults is done properly by 
        comparing the function result with one of the expected 
        results.""" 
        parent1 = np.copy(parents.genomes[0])
        parent2 = np.copy(parents.genomes[1])
        c = parents.chrlen
        children = parents.assortment().genomes
        assert \
        (children == np.append(parent1[:c], parent2[:c])).all() or\
        (children == np.append(parent2[:c], parent1[:c])).all() or\
        (children == np.append(parent1[:c], parent2[c:])).all() or\
        (children == np.append(parent2[:c], parent1[c:])).all() or\
        (children == np.append(parent1[c:], parent2[:c])).all() or\
        (children == np.append(parent2[c:], parent1[:c])).all() or\
        (children == np.append(parent1[c:], parent2[c:])).all() or\
        (children == np.append(parent2[c:], parent1[c:])).all()

    @pytest.mark.parametrize("mrate", [0, 0.3, 0.8, 1]) 
    def test_mutate_unbiased(self, spop, mrate):
        """Test that, in the absence of a +/- bias, the appropriate
        proportion of the genome is mutated."""
        genomes = np.copy(spop.genomes)
        spop.mutate(mrate,1)
        assert abs((1-np.mean(genomes == spop.genomes))-mrate) < 0.01

    @pytest.mark.parametrize("mratio", [0, 0.1, 0.5, 1])
    def test_mutate_biased(self, spop, mratio):
        """Test that the bias between positive and negative mutations is
        implemented correctly."""
        t = 0.01 # Tolerance
        mrate = 0.5 
        g0 = np.copy(spop.genomes)
        spop.mutate(mrate,mratio)
        g1 = spop.genomes
        is1 = (g0==1)
        is0 = np.logical_not(is1)
        assert abs((1-np.mean(g0[is1] == g1[is1]))-mrate) < t
        assert abs((1-np.mean(g0[is0] == g1[is0]))-mrate*mratio) < t 

    @pytest.mark.parametrize("sexvar",[True, False])
    @pytest.mark.parametrize("m",[0.0, 0.3, 0.8, 1.0])
    def test_growth(self,conf,sexvar,m):
        """Test number of children produced for all-adult population
        for sexual and asexual conditions."""
        # Make and grow population
        precision = 0.1
        n = 500
        params = conf.params.copy()
        params["sexual"] = sexvar
        params["age_random"] = False
        params["start_pop"] = n
        pop = Population(params, conf.genmap, np.array([-1]),
            np.array([[-1],[-1]]))
        pop.genomes = chance(m, pop.genomes.shape).astype(int)
        pop.growth(np.linspace(0,1,2*pop.nbase+1),1,0,0,1)
        # Calculate proportional observed and expected growth
        x = 2 if sexvar else 1
        obs_growth = (pop.N - n)/float(n)
        exp_growth = m/x
        assert abs(exp_growth-obs_growth) < precision

    @pytest.mark.parametrize("nparents",[1, 3, 5])
    def test_growth_smallpop(self, pop1, nparents):
        """Test that odd numbers of sexual parents are dropped and that a
        sexual parent population of size 1 doesn't reproduce."""
        parents = Population(pop1.params(), pop1.genmap, 
                pop1.ages[:nparents],pop1.genomes[:nparents])
        parents.sex = True
        parents.growth(np.linspace(0,1,2*parents.nbase+1),1,0,0,1)
        assert parents.N == (nparents + (nparents-1)/2)

    @pytest.mark.parametrize("sexvar",[True, False])
    def test_growth_extreme_starvation(self, spop, sexvar):
        """Confirm that growth() handles extreme starvation factors 
        correctly (probability limits at 0 and 1)."""
        pop0 = spop.clone()
        pop1 = spop.clone()
        pop2 = spop.clone()
        pop0.ages = pop1.ages = pop2.ages = np.tile(spop.maturity, spop.N)
        pop0.sex = pop1.sex = pop2.sex = sexvar
        exp_N = math.floor(1.5*spop.N) if sexvar else (2*spop.N)
        rrange = np.linspace(0.001, 1 , 2*spop.nbase+1)
        pop0.growth(rrange, 1e10, 0, 0, 1)
        pop1.growth(rrange, -1e10, 0, 0, 1)
        pop2.growth(rrange, 1e-10, 0, 0, 1)
        assert pop0.N == spop.N
        assert pop1.N == spop.N
        assert pop2.N == exp_N

#################
### 3: RECORD ###
#################

@pytest.mark.skipif(not runRecordTests, 
        reason="Not running Record tests.")
class TestRecord:
    """Test methods of the Record class."""

    # Initialisation

    def test_init_record(self, record, conf, pop1):
        r = record.record
        n = conf.number_of_snapshots
        m = conf.number_of_stages
        w = conf.window_size
        def sameshape(keys,ref):
            """Test whether all listed record arrays have identical 
            shape."""
            return np.all([r[x].shape == ref for x in keys])
        assert (r["genmap"] == pop1.genmap).all()
        assert (r["chr_len"] == np.array([pop1.chrlen])).all()
        assert (r["n_bases"] == np.array([pop1.nbase])).all()
        assert (r["max_ls"] == np.array([pop1.maxls])).all()
        assert (r["maturity"] == np.array([pop1.maturity])).all()
        assert (r["d_range"] == np.linspace(1,0,2*pop1.nbase+1)).all()
        assert (r["r_range"] == np.linspace(0,1,2*pop1.nbase+1)).all()
        assert (r["snapshot_stages"] == conf.snapshot_stages + 1).all()
        assert sameshape(["death_mean", "death_sd", "repr_mean",
                    "repr_sd", "fitness"], (n,pop1.maxls))
        assert sameshape(["density_surv", "density_repr"],
                    (n,2*pop1.nbase+1))
        assert sameshape(["entropy","junk_death","junk_repr",
                    "junk_fitness"], (n,))
        assert sameshape(["population_size", "resources", "surv_penf",
                    "repr_penf"], (m,))
        assert sameshape(["age_distribution"],(m,pop1.maxls))
        assert sameshape(["n1", "n1_std"], (n,pop1.chrlen))
        assert sameshape(["s1"], (n, pop1.chrlen - w + 1))

    # Per-stage updating

    def test_quick_update(self, record, pop1):
        """Test that every-stage update function records correctly."""
        record.quick_update(0, pop1, 100, 1, 1)
        r = record.record
        agedist=np.bincount(pop1.ages,minlength=pop1.maxls)/float(pop1.N)
        assert r["population_size"][0] == pop1.N
        assert r["resources"][0] == 100
        assert r["surv_penf"][0] == 1
        assert r["repr_penf"][0] == 1
        assert (r["age_distribution"][0] == agedist).all()

    def test_update_agestats(self,record,pop1):
        """Test if update_agestats properly calculates agestats for pop1
        (genomes filled with ones)."""
        record.update_agestats(pop1,0)
        r = record.record
        assert  np.isclose(r["death_mean"][0],
                np.tile(r["d_range"][-1],r["max_ls"])).all()
        assert np.isclose(r["death_sd"][0], np.zeros(r["max_ls"])).all()
        assert np.isclose(r["repr_mean"][0], 
                np.append(np.zeros(r["maturity"]),
                    np.tile(r["r_range"][-1],
                        r["max_ls"]-r["maturity"]))).all()
        assert np.isclose(r["repr_sd"][0], np.zeros(r["max_ls"])).all()
        assert r["density_surv"][0][-1] == 1
        assert r["density_repr"][0][-1] == 1

    def test_update_shannon_weaver_degenerate(self,record,pop1):
        """Test if equals zero when all set members are of same type."""
        assert record.update_shannon_weaver(pop1) == -0

    def test_update_shannon_weaver(self,record,spop,conf):
        """Test that shannon weaver entropy is computed correctly for
        a newly-initialised population."""
        precision = 0.015
        b = spop.nbase
        print spop.nbase, len(spop.genmap), conf.n_neutral
        print spop.params()

        props = np.array([spop.maxls, spop.maxls-spop.maturity, conf.n_neutral])\
                /float(len(spop.genmap)) # Expected proportion of genome
                # in survival loci, reproductive loci, etc.
        print props
        print "Sum of props = ", sum(props)
        probs = np.array([conf.g_dist[x] for x in ["s", "r", "n"]])
                # Probability of a 1 for each locus type
        print probs
        dists = np.array(
                [[comb(2*b, x)*p**x*(1-p)**(2*b-x) for x in np.arange(2*b+1)]\
                        for p in probs])
                # Binomial distribution values for 0 to 2*b zeros for each
        print "Sum, shape of dists = ", np.sum(dists,1), dists.shape
        exp = np.sum(dists * props[:,np.newaxis], 0) 
            # expected proportions of loci with each number of 1's over
            # entire genome
        print exp
        exp_entropy = st.entropy(exp)
        obs_entropy = record.update_shannon_weaver(spop)
        assert abs(exp_entropy - obs_entropy) < precision

    def test_sort_by_age(self, record):
        """Test if sort_by_age correctly sorts an artificial genome 
        array."""
        genmap = np.sort(record.record["genmap"])
        ix = np.arange(len(genmap))
            # Randomly reshuffle genmap
        np.random.shuffle(ix)
        record.record["genmap"] = genmap[ix]
        b = record.record["n_bases"]
        genome = np.tile(ix.reshape((len(ix),1)),b)
            # Make into a col vector
        genome = genome.reshape((1,len(genome)*b))[0]
            # Flatten col vector to get one element per genome bit
        mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),b)


    def test_update_invstats(self,record,pop1):
        """Test if update_invstats properly calculates genomestats for 
        pop1 (genomes filled with ones)."""
        record.update_invstats(pop1,0)
        r = record.record
        print np.sum(r["n1"][0] == np.ones(r["chr_len"]))
        assert (r["n1"][0] == np.ones(r["chr_len"])).all()
        assert (r["n1_std"][0] == np.zeros(r["chr_len"])).all()
        assert r["entropy"][0] == -0
        assert np.isclose(r["junk_death"][0], r["d_range"][-1])
        assert np.isclose(r["junk_repr"][0], r["r_range"][-1])

    def test_update(self,record,pop1):
        """Test that update properly chains quick_update,
        update_agestats and update_invstats."""
        record1 = copy.deepcopy(record)
        record.update(pop1, 100, 1, 1, 0, 0, False)
        record1.quick_update(0, pop1, 100, 1, 1)
        r = record.record
        r1 = record1.record
        for k in r.keys():
            assert (np.array(r[k]) == np.array(r1[k])).all()
        record.update(pop1, 100, 1, 1, 0, 0, True)
        record1.update_agestats(pop1, 0)
        record1.update_invstats(pop1, 0)
        r = record.record
        r1 = record1.record
        for k in r.keys():
            assert (np.array(r[k]) == np.array(r1[k])).all()
    
    # Test final updating

    def test_age_wise_n1(self,record):
        """Test if ten consecutive array items are correctly averaged."""
        genmap = record.record["genmap"]
        b = record.record["n_bases"]
        ix = np.arange(len(genmap))
        mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),b)
        mask = mask.reshape((1,len(mask)*b))[0]
        record.record["mask"] = np.array([mask])
        print mask.shape
        assert (record.age_wise_n1("mask") == ix).all()

    def test_actual_death_rate(self, record):
        """Test if actual_death_rate returns expected results for 
        artificial data."""
        r = record.record
        maxls = r["max_ls"][0]
        r["age_distribution"] = np.tile(1/float(maxls), (3, maxls))
        r["population_size"] = np.array([maxls*4,maxls*2,maxls])
        print r["age_distribution"]
        print r["population_size"]
        adr = record.actual_death_rate()
        print adr
        assert (adr[:,:-1] == 0.5).all()
        assert (adr[:,-1] == 1).all()

    def test_final_update(self, pop1, record, conf):
        s = conf.number_of_snapshots
        t = conf.number_of_stages
        for n in range(s):
            record.update(pop1, 100, 1, 1, 0, n, True)
        for m in range(t):
            record.update(pop1, 100, 1, 1, m, 0, False)
        record.final_update(conf.window_size)
        r = record.record
        # Predicted fitness array:
        pf = np.arange(r["max_ls"], dtype=float)-r["maturity"]+1
        pf[pf<0]=0
        # Predicted actual death rate:
        pad = np.append(np.ones(r["maturity"]-1), np.array([1-pop1.N]))
        pad = np.append(pad, np.ones(r["max_ls"] - len(pad)))
        assert (r["fitness"] == np.tile(pf, (s,1))).all()
        assert (r["age_wise_n1"] == 1).all()
        assert (r["age_wise_n1_std"] == 0).all()
        assert (r["junk_fitness"] == 1).all()
        assert (r["actual_death_rate"] == pad).all()
        assert (r["s1"] == 0).all()

@pytest.mark.skip(reason="New setup needed.")
@pytest.mark.xfail
def test_post_cleanup():
    """Kill tempfiles made for test. Not really a test at all."""
    os.remove("sample_pop.txt")
    os.remove("sample_rec.txt")
    os.remove("log.txt")
