# TODO: 
# - Test repr_penf against surv_penf; bug from 4-Jul-2016
# - Speed up tests involving get_startpop (currently very slow)

import pyximport; pyximport.install()
from gs_core import Simulation, Run, Outpop, Population, Record, Config # Classes
from gs_core import chance, get_runtime, execute_run, testage, testgen # Functions
import pytest, random, string, subprocess, math, copy, os, sys, cPickle, datetime
import numpy as np
import scipy.stats as st
from scipy.misc import comb

runFunctionConfigTests=True # Works with new setup
runPopulationTests=True # "
runRecordTests=True # "
runRunTests=True # "
runSimulationTests=True

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
    S = Simulation("config_test", "", -1, 10, False)
    c = copy.deepcopy(S.conf)
    c.number_of_stages = 100
    if request.param == "random":
        # Randomise fundamental parameters
        c.g_dist_s,c.g_dist_r,c.g_dist_n = [random.random() for x in range(3)]
        db_low, rb_low = np.random.uniform(size=2)
        db_high = db_low + random.uniform(0, 1-db_low)
        rb_high = rb_low + random.uniform(0, 1-rb_low)
        c.death_bound,c.repr_bound = [db_low, db_high],[rb_low, rb_high]
        c.r_rate, c.m_rate, c.m_ratio = [random.random() for x in range(3)]
        c.max_ls = random.randint(20, 99)
        c.maturity = random.randint(5, c.max_ls-2)
        #c.n_neutral = random.randint(1, 100)
        c.n_base = random.randint(5, 25)
        c.surv_pen = random.choice([True, False])
        c.repr_pen = random.choice([True, False])
        c.death_inc = random.randint(1, 10)
        c.repr_dec = random.randint(1, 10)
        gm_len = c.max_ls + (c.max_ls - c.maturity) + c.n_neutral
        c.window_size = random.randint(1, gm_len*c.n_base)
        c.generate()
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
def record(request,conf):
    """Create a record from pop1 as defined in configuration file."""
    spaces = 2 * conf.n_base + 1
    c = copy.deepcopy(conf)
    c.d_range,c.r_range = np.linspace(1,0,spaces),np.linspace(0,1,spaces)
    rec = Record(c)
    return rec

@pytest.fixture()
def run(request,conf):
    """Create an unseeded run object from configuration."""
    return Run(conf, "", 0, 100, False)

@pytest.fixture()
def simulation(request,conf):
    """Create an unseeded simulation object from configuration."""
    sim = Simulation("config_test", "", -1, 100, False)
    sim.conf = conf
    sim.runs = [Run(sim.conf, sim.startpop[0], sim.report_n,
                sim.verbose) for n in xrange(sim.conf.number_of_runs)]
    return sim

# Create separate fixtures to avoid unnecessary trebling of tests that
# don't depend on config state
@pytest.fixture(scope="session")
def S(request):
    """Create an unmodified, unseeded simulation object for procedure testing."""
    return Simulation("config_test", "", -1, 100, False)
@pytest.fixture()
def R(request, S):
    """Create an unmodified, unseeded run object for procedure testing."""
    return Run(S.conf, "", 0, 100, False)
@pytest.fixture()
def P(request, R):
    return R.population.toPop()
@pytest.fixture()
def Rc(request, R):
    return R.record

####################
### 0: DUMMY RUN ###
####################

def test_sim_run():
    # Begin by running a dummy simulation and saving the output
    # Also functions as test of output functions
    scriptdir = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(scriptdir)
    subprocess.call(["python", "genome_simulation.py", "."])
    os.rename("output.sim", "sample_output.sim")
    os.rename("output.rec", "sample_output.rec")
    os.rename("log.txt", "sample_log.txt")

@pytest.fixture(scope="session")
def run_sim(request):
    """Get pre-run simulation created during test_sim_run."""
    try:
        f = open("sample_output.sim", "rb")
        px = cPickle.load(f)
        return px
    finally:
        f.close()

@pytest.mark.skipif(not runFunctionConfigTests,
        reason="Not running function/Config tests.")
class TestFunctionsConfig:

    @pytest.mark.parametrize("p", [0,1])
    def test_chance_degenerate(self, p):
        """Tests wether p=1 returns True/1 and p=0 returns False/0."""
        shape=(1000,1000)
        assert np.all(chance(p, shape).astype(int) == p)

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_chance(self, p):
        precision = 0.01
        """Test that the shape of the output is correct and that the mean
        over many trials is close to the expected value."""
        shape=(random.randint(1,1000),random.randint(1,1000))
        c = chance(p, shape)
        s = c.shape
        assert c.shape == shape and c.dtype == "bool"
        assert abs(p-np.mean(c)) < precision

    def test_get_runtime(self):
        """Test that get_runtime calculates differences correctly."""
        a = datetime.datetime(1, 1, 1, 0, 0, 0, 0)
        b = datetime.datetime(1, 1, 2, 0, 0, 5, 0)
        c = datetime.datetime(1, 1, 3, 1, 0, 20, 0)
        d = datetime.datetime(1, 1, 3, 2, 5, 0, 0)
        B = get_runtime(a, b)
        assert B == "Total runtime: 1 days, 5 seconds."
        C = get_runtime(a, c)
        assert C == "Total runtime: 2 days, 1 hours, 20 seconds."
        D = get_runtime(a, d)
        assert D == "Total runtime: 2 days, 2 hours, 5 minutes, 0 seconds."

    def test_execute_run(self, R):
        run1, run2 = copy.deepcopy(R), copy.deepcopy(R)
        xr1 = execute_run(run1, 100)
        assert xr1.complete
        assert not xr1.dieoff
        r2p = run2.population
        r2p.ages = np.array([r2p.maxls-1] * r2p.N)
        maxfail = random.randrange(9)+1
        print run2, maxfail
        xr2 = execute_run(run2, maxfail)
        assert xr2.complete
        assert xr2.dieoff
        assert xr2.record.record["percent_dieoff"] == 100.0
        assert xr2.record.record["prev_failed"] == maxfail - 1

    @pytest.mark.parametrize("sexvar,nsnap", [(True, 10), (False, 0.1)])
    def test_config_generate(self, conf, sexvar, nsnap):
        """Test that gen_conf correctly generates derived simulation params."""
        # Remove stuff that gets introduced/changed in gen_conf
        c = copy.deepcopy(conf)
        del c.g_dist, c.genmap, c.chr_len, c.d_range, c.r_range, c.params
        del c.snapshot_stages
        if c.sexual: c.repr_bound[1] /= 2
        crb1 = c.repr_bound[1]
        # Set parameters and run
        c.number_of_snapshots = nsnap
        c.sexual = sexvar
        c.generate()
        # Test output
        assert c.g_dist["s"] == c.g_dist_s
        assert c.g_dist["r"] == c.g_dist_r
        assert c.g_dist["n"] == c.g_dist_n
        assert len(c.genmap) == c.max_ls + (c.max_ls-c.maturity) +\
                c.n_neutral
        assert c.chr_len == len(c.genmap) * c.n_base
        assert c.repr_bound[1]/crb1 == 2 if sexvar else 1
        assert np.array_equal(c.d_range, np.linspace(c.death_bound[1], 
            c.death_bound[0],2*c.n_base+1))
        assert np.array_equal(c.r_range, np.linspace(c.repr_bound[0], 
            c.repr_bound[1],2*c.n_base+1))
        assert len(c.snapshot_stages) == c.number_of_snapshots if \
                type(nsnap) is int else int(nsnap * c.number_of_stages)
        assert np.array_equal(c.snapshot_stages,np.around(np.linspace(
            0, c.number_of_stages-1, c.number_of_snapshots), 0))
        assert c.params["sexual"] == sexvar
        assert c.params["chr_len"] == c.chr_len
        assert c.params["n_base"] == c.n_base
        assert c.params["maturity"] == c.maturity
        assert c.params["max_ls"] == c.max_ls
        assert c.params["age_random"] == c.age_random
        assert c.params["start_pop"] == c.start_pop
        assert c.params["g_dist"] == c.g_dist

######################
### 2: POPULATION  ###
######################

@pytest.mark.skipif(not runPopulationTests,
        reason="Not running Population tests.")
class TestPopulationClass:
    """Test population object methods."""

    # Initialisation
    def test_init_population_blank(self, conf):
        """Test that population parameters are correct for random and
        nonrandom ages."""
        precision = 1.1
        x = random.random()
        conf1 = copy.deepcopy(conf)
        conf1.params["start_pop"] = 2000
        conf1.params["age_random"] = False
        conf1.params["g_dist"] = {"s":x,"r":x,"n":x}
        pop_a = Population(conf1.params, conf1.genmap, np.array([-1]),
            np.array([[-1],[-1]]))
        conf1.params["age_random"] = True
        pop_b = Population(conf1.params, conf1.genmap, np.array([-1]),
            np.array([[-1],[-1]]))
        print np.mean(pop_b.ages)
        print abs(np.mean(pop_b.ages)-pop_b.maxls/2)
        for pp in [pop_a, pop_b]:
            assert pp.sex == conf1.params["sexual"]
            assert pp.chrlen == conf1.params["chr_len"]
            assert pp.nbase == conf1.params["n_base"]
            assert pp.maxls == conf1.params["max_ls"]
            assert pp.maturity == conf1.params["maturity"]
            assert pp.N == conf1.params["start_pop"]
            assert np.array_equal(pp.genmap, conf1.genmap)
            assert abs(np.mean(pp.genomes)-x) < 0.05
        assert np.all(pop_a.ages == pop_a.maturity)
        assert not np.all(pop_b.ages == pop_b.maturity)
        assert abs(np.mean(pop_b.ages)-pop_b.maxls/2) < precision

    def test_init_population_nonblank(self, conf):
        """Test that population is generated correctly when ages and/or
        genomes are provided."""
        conf1 = copy.deepcopy(conf)
        x = random.random()
        ages = np.random.randint(0, 10, conf1.params["start_pop"])
        genomes = chance(1-x**2, 
                (conf1.params["start_pop"],conf1.params["chr_len"])).astype(int)
        conf1.params["age_random"] = False
        x = random.random()
        conf1.params["g_dist"] = {"s":x,"r":x,"n":x}
        pop_a = Population(conf1.params, conf1.genmap, ages, genomes)
        pop_b = Population(conf1.params, conf1.genmap, ages, testgen())
        pop_c = Population(conf1.params, conf1.genmap, testage(), genomes)
        # Check params
        for pp in [pop_a, pop_b, pop_c]:
            assert pp.sex == conf1.params["sexual"]
            assert pp.chrlen == conf1.params["chr_len"]
            assert pp.nbase == conf1.params["n_base"]
            assert pp.maxls == conf1.params["max_ls"]
            assert pp.maturity == conf1.params["maturity"]
            assert pp.N == conf1.params["start_pop"]
        # Check ages
        assert np.array_equal(ages, pop_a.ages)
        assert np.array_equal(ages, pop_b.ages)
        assert not np.array_equal(ages, pop_c.ages)
        assert np.all(pop_c.ages == conf1.params["maturity"])
        # Check genomes
        assert np.array_equal(genomes, pop_a.genomes)
        assert np.array_equal(genomes, pop_c.genomes)
        assert not np.array_equal(genomes, pop_b.genomes)
        assert abs(np.mean(pop_b.genomes) - x) < 0.05

    def test_popgen_independence(self, P):
        """Test that generating a population from another and then manipulating
        it cloned population does not affect original (important for 
        reproduction)."""
        P1 = P.clone()
        P2 = Population(P1.params(), P1.genmap, P1.ages, P1.genomes)
        P3 = Population(P1.params(), P1.genmap, P1.ages[:100], P1.genomes[:100])
        P4 = Population(P3.params(), P3.genmap, P3.ages, P3.genomes)
        P2.mutate(0.5, 1)
        P4.mutate(0.5, 1)
        # Test ages
        assert np.array_equal(P.ages, P1.ages)
        assert np.array_equal(P.ages, P2.ages)
        assert np.array_equal(P.ages[:100], P3.ages)
        assert np.array_equal(P.ages[:100], P4.ages)
        # Test genomes
        assert np.array_equal(P.genomes, P1.genomes)
        assert np.array_equal(P.genomes[:100], P3.genomes)
        assert P.genomes.shape == P2.genomes.shape
        assert P.genomes[:100].shape == P4.genomes.shape
        assert not np.array_equal(P.genomes, P2.genomes)
        assert not np.array_equal(P.genomes[:100], P4.genomes)

    # Minor methods
    # Genome array
    """Test that new genome arrays are generated correctly."""
    genmap_simple = np.append(np.arange(25),
            np.append(np.arange(24)+100, 200))
    genmap_shuffled = np.copy(genmap_simple)
    random.shuffle(genmap_shuffled)

    @pytest.mark.parametrize("gd", [
        {"s":random.random(), "r":random.random(), "n":random.random()},
        {"s":random.random(), "r":random.random(), "n":random.random()}])
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

    def test_shuffle(self, P):
        """Test if all ages, therefore individuals, present before the
        shuffle are also present after it."""
        P2 = P.clone() # clone tested separately
        P2.shuffle()
        is_shuffled = \
                not np.array_equal(P.genomes, P2.genomes)
        P.ages.sort()
        P2.ages.sort()
        assert is_shuffled
        assert np.array_equal(P.ages, P2.ages)

    def test_clone(self, P):
        """Test if cloned population is identical to parent population,
        by comparing params, ages, genomes."""
        P2 = P.clone()
        assert P.params() == P2.params()
        assert (P.genmap == P2.genmap).all()
        assert (P.ages == P2.ages).all()
        assert (P.genomes == P2.genomes).all()

    def test_increment_ages(self, P):
        """Test if all ages are incrementd by one."""
        P2 = P.clone()
        P2.increment_ages()
        assert (P.ages+1 == P2.ages).all()

    def test_params(self, P):
        """Test that params returns (at least) the
        required information."""
        p = P.params()
        assert len(p) >= 5 and p["sexual"] == P.sex
        assert p["chr_len"] == P.chrlen
        assert p["n_base"] == P.nbase
        assert p["max_ls"] == P.maxls
        assert p["maturity"] == P.maturity

    def test_addto(self, P):
        """Test if a population is successfully appended to the receiver
        population, which remains unchanged, by appending a population to
        itself."""
        pop_a = P.clone()
        pop_b = P.clone()
        pop_b.addto(pop_a)
        assert np.array_equal(pop_b.ages, np.tile(pop_a.ages,2))
        assert np.array_equal(pop_b.genomes, np.tile(pop_a.genomes,(2,1)))
        assert pop_b.N == 2*pop_a.N

    # Death and crisis

    @pytest.mark.parametrize("p", [0, random.random(), 1])
    @pytest.mark.parametrize("adults_only,offset",[(False,0),(True,100)])
    def test_get_subpop_allsame(self, spop, p, adults_only, offset):
        precision = 0.1
        """Test if the probability of passing is close to that indicated
        by the genome (when all loci have the same distribution)."""
        pop = spop.clone()
        # Set appropriate loci to given probability value
        loci = np.nonzero(
                np.logical_and(pop.genmap >= offset, pop.genmap < offset + 100)
                )[0]
        pos = np.array([range(pop.nbase) + y for y in loci*pop.nbase])
        pos = np.append(pos, pos + pop.chrlen)
        min_age = pop.maturity if adults_only else 0
        pop.genomes[:,pos] = chance(p, pop.genomes[:,pos].shape).astype(int)
        # Get subpop and test
        subpop = pop.get_subpop(min_age, pop.maxls, offset,
                np.linspace(0,1,2*pop.nbase + 1))
        assert abs(np.mean(subpop) - p) < precision
        #assert abs(np.sum(subpop)/float(pop.N) - p)*(1-min_age/pop.maxls) < \
        #        precision

    def test_get_subpop_different(self, run):
        """Test whether individuals with different genotypes are selected
        with appropriately different frequencies by get_subpop."""
        precision = 0.1
        pop = run.population.clone().toPop()
        pop.ages = np.array([pop.maturity+1]*300)
        pop.N = 300
        x = np.zeros((100, pop.genomes.shape[1]))
        pop.genomes = np.vstack((x,np.ones(x.shape),
            chance(0.5,x.shape))).astype(int)
        subpop = pop.get_subpop(0, pop.maxls, 0,
                np.linspace(0,1,2*pop.nbase+1))
        assert abs(np.mean(subpop[:100])) < precision
        assert abs(np.mean(subpop[100:200]) - 1) < precision
        assert abs(np.mean(subpop[200:300]) - 0.5) < precision

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

    @pytest.mark.parametrize("p", [0.0, random.random(), 1.0])
    @pytest.mark.parametrize("x", [1.0, 3.0, 9.0])
    def test_death(self, spop, p, x):
        """Test if self.death() correctly inverts death probabilities
        and incorporates starvation factor to get survivor probabilities
        and survivor array."""
        precision = 0.15
        pop = spop.clone()
        # Modify survival loci to specified p
        b = pop.nbase
        surv_loci = np.nonzero(spop.genmap<100)[0]
        surv_pos = np.array([range(b) + y for y in surv_loci*b])
        surv_pos = np.append(surv_pos, surv_pos + pop.chrlen)
        pop.genomes[:, surv_pos] =\
                chance(p, pop.genomes[:, surv_pos].shape).astype(int)
        # Call and test death function
        pop2 = pop.clone()
        print pop2.genomes[:, surv_pos]
        pop2.death(np.linspace(1,0,2*pop2.nbase+1), x)
        pmod = max(0, min(1, (1-x*(1-p))))
        assert abs(pop2.N/float(pop.N) - pmod) < precision

    def test_death_extreme_starvation(self, P):
        """Confirm that death() handles extreme starvation factors
        correctly (probability limits at 0 and 1)."""
        pop0 = P.clone()
        pop1 = P.clone()
        pop2 = P.clone()
        drange = np.linspace(1, 0.001, 2*P.nbase+1)
        pop0.death(drange, 1e10)
        pop1.death(drange, -1e10)
        pop2.death(drange, 1e-10)
        assert pop0.N == 0
        assert pop1.N == P.N
        assert pop2.N == P.N

    @pytest.mark.parametrize("p", [0, random.random(), 1])
    def test_crisis(self, P, p):
        """Test whether extrinsic death crisis removes the expected
        fraction of the population."""
        precision = 0.1
        pop = P.clone()
        pop.crisis(p)
        assert abs(pop.N - p*P.N)/P.N < precision

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

    def test_recombine_float(self, spop):
        """Test if genome changes if recombination fn.chance p; 0<p<1."""
        pop = spop.clone()
        pop.recombine(np.random.uniform(0.01,0.09))
        assert not np.array_equal(pop.genomes, spop.genomes)

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

    @pytest.mark.parametrize("mrate", [0, random.random(), 1])
    def test_mutate_unbiased(self, P, mrate):
        """Test that, in the absence of a +/- bias, the appropriate
        proportion of the genome is mutated."""
        P1 = P.clone()
        t = 0.001 # Tolerance
        genomes = np.copy(P1.genomes)
        P1.mutate(mrate,1)
        assert abs((1-np.mean(genomes == P1.genomes))-mrate) < t

    @pytest.mark.parametrize("mratio", [0, random.random(), 1])
    def test_mutate_biased(self, P, mratio):
        """Test that the bias between positive and negative mutations is
        implemented correctly."""
        t = 0.0015 # Tolerance
        P1 = P.clone()
        mrate = 0.5
        g0 = np.copy(P1.genomes)
        P1.mutate(mrate,mratio)
        g1 = P1.genomes
        is1 = (g0==1)
        is0 = np.logical_not(is1)
        assert abs(np.mean(g0[is1] != g1[is1])-mrate) < t
        assert abs(np.mean(g0[is0] != g1[is0])-mrate*mratio) < t

    @pytest.mark.parametrize("x", [1.0, 5.0])
    @pytest.mark.parametrize("sexvar",[True, False])
    def test_growth(self, spop, x, sexvar):
        """Test if self.death() correctly inverts death probabilities
        and incorporates starvation factor to get survivor probabilities
        and survivor array."""
        p = random.random()
        precision = 0.05
        pop = spop.clone()
        pop.sex = sexvar
        # Modify reproductive loci to specified p
        b = pop.nbase
        repr_loci = np.nonzero(
                np.logical_and(spop.genmap>=100, spop.genmap<200))[0]
        print repr_loci
        repr_pos = np.array([range(b) + y for y in repr_loci*b])
        repr_pos = np.append(repr_pos, repr_pos + pop.chrlen)
        pop.genomes[:, repr_pos] =\
                chance(p, pop.genomes[:, repr_pos].shape).astype(int)
        assert abs(np.mean(pop.genomes[:, repr_pos]) - p) < precision # Validate
        # Call and test death function
        pop2 = pop.clone()
        print pop2.genomes[:, repr_pos]
        pop2.growth(np.linspace(0,1,2*pop2.nbase+1), x, 0, 0, 1)
        # Calculate proportional observed and expected growth
        z = x*2 if sexvar else x
        obs_growth = (pop2.N - pop.N)/float(pop.N)
        exp_growth = p/z
        print pop.N, pop2.N, p, z
        assert abs(exp_growth-obs_growth) < precision

    @pytest.mark.parametrize("nparents",[1, 3, 5])
    def test_growth_smallpop(self, P, nparents):
        """Test that odd numbers of sexual parents are dropped and that a
        sexual parent population of size 1 doesn't reproduce."""
        parents = Population(P.params(), P.genmap, P.ages[:nparents],
                np.ones(P.genomes.shape,dtype=int)[:nparents])
        parents.sex = True
        parents.growth(np.linspace(0,1,2*parents.nbase+1),1,0,0,1)
        assert parents.N == (nparents + (nparents-1)/2)

    @pytest.mark.parametrize("sexvar",[True, False])
    def test_growth_extreme_starvation(self, P, sexvar):
        """Confirm that growth() handles extreme starvation factors
        correctly (probability limits at 0 and 1)."""
        pop0 = P.clone()
        pop1 = P.clone()
        pop2 = P.clone()
        pop0.ages = pop1.ages = pop2.ages = np.tile(P.maturity, P.N)
        pop0.sex = pop1.sex = pop2.sex = sexvar
        exp_N = math.floor(1.5*P.N) if sexvar else (2*P.N)
        rrange = np.linspace(0.001, 1 , 2*P.nbase+1)
        pop0.growth(rrange, 1e10, 0, 0, 1)
        pop1.growth(rrange, -1e10, 0, 0, 1)
        pop2.growth(rrange, 1e-10, 0, 0, 1)
        assert pop0.N == P.N
        assert pop1.N == P.N
        assert pop2.N == exp_N

#################
### 3: RECORD ###
#################

@pytest.mark.skipif(not runRecordTests,
        reason="Not running Record tests.")
class TestRecordClass:
    """Test methods of the Record class."""

    # Initialisation

    def test_init_record(self, record, conf):
        r = record.record
        n = conf.number_of_snapshots
        m = conf.number_of_stages
        w = conf.window_size
        def assert_sameshape(keys,ref):
            """Test whether all listed record arrays have identical
            shape."""
            for x in keys:
                assert r[x].shape == ref
                assert np.all(r[x] == 0)
        assert (r["genmap"] == conf.genmap).all()
        assert (r["chr_len"] == np.array([conf.chr_len])).all()
        assert (r["n_bases"] == np.array([conf.n_base])).all()
        assert (r["max_ls"] == np.array([conf.max_ls])).all()
        assert (r["maturity"] == np.array([conf.maturity])).all()
        assert (r["d_range"] == np.linspace(1,0,2*conf.n_base+1)).all()
        assert (r["r_range"] == np.linspace(0,1,2*conf.n_base+1)).all()
        assert (r["snapshot_stages"] == conf.snapshot_stages).all()
        assert_sameshape(["death_mean", "death_sd", "repr_mean",
                    "repr_sd", "age_wise_fitness_product",
                    "junk_age_wise_fitness_product",
                    "age_wise_fitness_contribution",
                    "junk_age_wise_fitness_contribution"], (n,conf.max_ls))
        assert_sameshape(["density_surv", "density_repr"],
                    (n,2*conf.n_base+1))
        assert_sameshape(["entropy","junk_death","junk_repr","junk_fitness",
                    "fitness"], (n,))
        assert_sameshape(["population_size", "resources", "surv_penf",
                    "repr_penf"], (m,))
        assert_sameshape(["age_distribution"],(m,conf.max_ls))
        assert_sameshape(["n1", "n1_std"], (n,conf.chr_len))
        assert_sameshape(["s1"], (n, conf.chr_len - w + 1))

    # Per-stage updating

    def test_quick_update(self, Rc, P):
        """Test that every-stage update function records correctly."""
        Rc2 = copy.deepcopy(Rc)
        Rc2.quick_update(0, P, 100, 1, 1)
        r = Rc2.record
        agedist=np.bincount(P.ages,minlength=P.maxls)/float(P.N)
        assert r["population_size"][0] == P.N
        assert r["resources"][0] == 100
        assert r["surv_penf"][0] == 1
        assert r["repr_penf"][0] == 1
        assert (r["age_distribution"][0] == agedist).all()

    def test_update_agestats(self,record,pop1):
        """Test if update_agestats properly calculates agestats for pop1
        (genomes filled with ones)."""
        rec = copy.deepcopy(record)
        rec.update_agestats(pop1,0)
        r = rec.record
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

    def test_update_shannon_weaver_degenerate(self,Rc,P):
        """Test if equals zero when all set members are of same type."""
        P2 = P.clone()
        P2.genomes = np.ones(P2.genomes.shape, dtype=int)
        assert Rc.update_shannon_weaver(P2) == -0

    def test_update_shannon_weaver(self,record,spop,conf):
        """Test that shannon weaver entropy is computed correctly for
        a newly-initialised population."""
        precision = 0.015
        b = spop.nbase
        props = np.array([spop.maxls, spop.maxls-spop.maturity, conf.n_neutral])\
                /float(len(spop.genmap)) # Expected proportion of genome
                # in survival loci, reproductive loci, etc.
        probs = np.array([conf.g_dist[x] for x in ["s", "r", "n"]])
                # Probability of a 1 for each locus type
        dists = np.array(
                [[comb(2*b, x)*p**x*(1-p)**(2*b-x) for x in np.arange(2*b+1)]\
                        for p in probs])
                # Binomial distribution values for 0 to 2*b zeros for each
        exp = np.sum(dists * props[:,np.newaxis], 0)
            # expected proportions of loci with each number of 1's over
            # entire genome
        exp_entropy = st.entropy(exp)
        obs_entropy = record.update_shannon_weaver(spop)
        assert abs(exp_entropy - obs_entropy) < precision

    def test_sort_by_age(self, record):
        """Test if sort_by_age correctly sorts an artificial genome
        array."""
        rec = copy.deepcopy(record)
        genmap = np.sort(rec.record["genmap"])
        ix = np.arange(len(genmap))
            # Randomly reshuffle genmap
        np.random.shuffle(ix)
        rec.record["genmap"] = genmap[ix]
        b = rec.record["n_bases"]
        genome = np.tile(ix.reshape((len(ix),1)),b)
            # Make into a col vector
        genome = genome.reshape((1,len(genome)*b))[0]
            # Flatten col vector to get one element per genome bit
        mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),b)
        mask = mask.reshape((1,len(mask)*b))[0]
        np.set_printoptions(threshold=np.inf)
        print mask
        print genome
        print record.sort_by_age(genome)
        assert (rec.sort_by_age(genome).astype(int) == mask).all()

    def test_update_invstats(self,record,pop1):
        """Test if update_invstats properly calculates genomestats for
        pop1 (genomes filled with ones)."""
        rec = copy.deepcopy(record)
        rec.update_invstats(pop1,0)
        r = rec.record
        print np.sum(r["n1"][0] == np.ones(r["chr_len"]))
        assert (r["n1"][0] == np.ones(r["chr_len"])).all()
        assert (r["n1_std"][0] == np.zeros(r["chr_len"])).all()
        assert r["entropy"][0] == -0
        assert np.isclose(r["junk_death"][0], r["d_range"][-1])
        assert np.isclose(r["junk_repr"][0], r["r_range"][-1])

    def test_update(self,Rc,P):
        """Test that update properly chains quick_update,
        update_agestats and update_invstats."""
        Rc1,Rc2 = copy.deepcopy(Rc), copy.deepcopy(Rc)
        Rc1.update(P, 100, 1, 1, 0, 0, False)
        Rc2.quick_update(0, P, 100, 1, 1)
        r1,r2 = Rc1.record, Rc2.record
        for k in r1.keys():
            assert (np.array(r1[k]) == np.array(r2[k])).all()
        Rc1.update(P, 100, 1, 1, 0, 0, True)
        Rc2.update_agestats(P, 0)
        Rc2.update_invstats(P, 0)
        r1,r2 = Rc1.record, Rc2.record
        for k in r1.keys():
            assert (np.array(r1[k]) == np.array(r2[k])).all()

    # Test final updating

    def test_age_wise_n1(self,record):
        """Test if ten consecutive array items are correctly averaged."""
        rec = copy.deepcopy(record)
        genmap = rec.record["genmap"]
        b = rec.record["n_bases"]
        ix = np.arange(len(genmap))
        mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),b)
        mask = mask.reshape((1,len(mask)*b))[0]
        rec.record["mask"] = np.array([mask])
        print mask.shape
        assert (rec.age_wise_n1("mask") == ix).all()

    def test_actual_death_rate(self, Rc):
        """Test if actual_death_rate returns expected results for
        artificial data."""
        Rc1 = copy.deepcopy(Rc)
        r = Rc1.record
        maxls = r["max_ls"][0]
        r["age_distribution"] = np.tile(1/float(maxls), (3, maxls))
        r["population_size"] = np.array([maxls*4,maxls*2,maxls])
        print r["age_distribution"]
        print r["population_size"]
        adr = Rc1.actual_death_rate()
        print adr
        assert (adr[:,:-1] == 0.5).all()
        assert (adr[:,-1] == 1).all()

    def test_final_update(self, pop1, record, conf):
        rec = copy.deepcopy(record)
        s = conf.number_of_snapshots
        t = conf.number_of_stages
        for n in range(s):
            rec.update(pop1, 100, 1, 1, 0, n, True)
        for m in range(t):
            rec.update(pop1, 100, 1, 1, m, 0, False)
        rec.final_update(conf.window_size)
        r = rec.record
        # Predicted age_wise_fitness_contribution array:
        pf = np.append(np.zeros((s,r["maturity"])),
                np.ones((s,r["max_ls"]-r["maturity"])),1)
        # Predicted actual death rate:
        pad = np.append(np.ones(r["maturity"]-1), np.array([1-pop1.N]))
        pad = np.append(pad, np.ones(r["max_ls"] - len(pad)))
        # All fitness terms should be positive
        assert (r["age_wise_fitness_contribution"] == pf).all()
        assert (r["junk_age_wise_fitness_contribution"] == pf).all()
        assert (r["age_wise_fitness_product"] == pf).all()
        assert (r["junk_age_wise_fitness_product"] == pf).all()
        assert (r["fitness"] == np.sum(pf,1)).all()
        assert (r["junk_fitness"] == np.sum(pf,1)).all()
        assert (r["age_wise_n1"] == 1).all()
        assert (r["age_wise_n1_std"] == 0).all()
        assert (r["actual_death_rate"] == pad).all()
        assert (r["s1"] == 0).all()

    def test_fitness_calc(self, run):
        """Test whether fitness is calculated correctly for various cases."""
        rec, p = copy.deepcopy(run.record), copy.deepcopy(run.population)
        r,s,t = rec.record,run.conf.number_of_snapshots,run.conf.number_of_stages
        r["sexual"] = False
        r["d_range"], r["r_range"] = [np.linspace(1,0,2*p.nbase+1)]*2
        for n in range(s):
            rec.update(p, 100, 1, 1, 0, n, True)
        for m in range(t):
            rec.update(p, 100, 1, 1, m, 0, False)
        rec1 = copy.deepcopy(rec); r1 = rec1.record; r1["sexual"]= True
        def gen_death(x): return np.tile(x, (p.N, p.maxls))
        def gen_repr(x): return np.hstack([np.zeros((p.N, p.maturity)),
            np.tile(x, (p.N, p.maxls-p.maturity))])
        def set_means_and_calc(x, y):
            r["death_mean"],r["repr_mean"] = gen_death(x), gen_repr(y)
            r1["death_mean"],r1["repr_mean"] = gen_death(x), gen_repr(y)
            rec.final_update(100)
            rec1.final_update(100)
        def assert_fitnesses(a, b):
            fc,fp = "age_wise_fitness_contribution","age_wise_fitness_product"
            assert np.array_equal(r["fitness"], np.sum(r[fc],1))
            assert np.all(r[fp][:,:p.maturity]==0)
            assert np.all(r[fc][:,:p.maturity]==0)
            assert np.all(np.isclose(r[fp][:,p.maturity:],a))
            assert np.all(np.isclose(r[fc][:,p.maturity:],b))
            for k in ["fitness", "age_wise_fitness_product",
                    "age_wise_fitness_contribution"]:
                assert np.all(r[k] >= 0)
                assert np.all(np.isclose(r[k]/2.0, r1[k]))
        # Certain reproduction, impossible death
        set_means_and_calc(0.0, 1.0)
        assert_fitnesses(1.0, 1.0)
        # Certain reproduction, random death
        death_rate = random.random()
        set_means_and_calc(death_rate, 1.0)
        assert_fitnesses(1-death_rate,
                (1-death_rate)**(np.arange(p.maturity,p.maxls)+1)) 
        # Random reproduction, zero death
        repr_rate = random.random()
        set_means_and_calc(0.0,repr_rate)
        assert_fitnesses(repr_rate,repr_rate) 
        # Random reproduction, certain death
        repr_rate = random.random()
        set_means_and_calc(1.0,repr_rate)
        assert_fitnesses(0,0)
        # Zero reproduction, random death
        death_rate = random.random()
        set_means_and_calc(death_rate,0.0)
        assert_fitnesses(0,0) 

    def test_fitness_calc_junk(self, run):
        """Same as test_fitness_calc, but for junk fitness."""
        rec, p = copy.deepcopy(run.record), copy.deepcopy(run.population)
        r,s,t = rec.record,run.conf.number_of_snapshots,run.conf.number_of_stages
        r["sexual"] = False
        r["d_range"], r["r_range"] = [np.linspace(1,0,2*p.nbase+1)]*2
        for n in range(s):
            rec.update(p, 100, 1, 1, 0, n, True)
        for m in range(t):
            rec.update(p, 100, 1, 1, m, 0, False)
        rec1 = copy.deepcopy(rec); r1 = rec1.record; r1["sexual"]= True
        def gen_death(x): return np.tile(x, r["junk_death"].shape)
        def gen_repr(x): return np.tile(x, r["junk_repr"].shape)
        def set_means_and_calc(x, y):
            r["junk_death"],r["junk_repr"] = gen_death(x), gen_repr(y)
            r1["junk_death"],r1["junk_repr"] = gen_death(x), gen_repr(y)
            rec.final_update(100)
            rec1.final_update(100)
        def assert_fitnesses(a, b):
            fc = "junk_age_wise_fitness_contribution"
            fp = "junk_age_wise_fitness_product"
            assert np.array_equal(r["junk_fitness"], np.sum(r[fc],1))
            assert np.all(r[fp][:,:p.maturity]==0)
            assert np.all(r[fc][:,:p.maturity]==0)
            assert np.all(np.isclose(r[fp][:,p.maturity:],a))
            assert np.all(np.isclose(r[fc][:,p.maturity:],b))
            for k in ["junk_fitness", "junk_age_wise_fitness_product",
                    "junk_age_wise_fitness_contribution"]:
                print k
                assert np.all(r[k] >= 0)
                assert np.all(np.isclose(r[k]/2.0, r1[k]))
        # Certain reproduction, impossible death
        set_means_and_calc(0.0, 1.0)
        assert_fitnesses(1.0, 1.0)
        # Certain reproduction, random death
        death_rate = random.random()
        set_means_and_calc(death_rate, 1.0)
        assert_fitnesses(1-death_rate,
                (1-death_rate)**(np.arange(p.maturity,p.maxls)+1)) 
        # Random reproduction, zero death
        repr_rate = random.random()
        set_means_and_calc(0.0,repr_rate)
        assert_fitnesses(repr_rate,repr_rate) 
        # Random reproduction, certain death
        repr_rate = random.random()
        set_means_and_calc(1.0,repr_rate)
        assert_fitnesses(0,0)
        # Zero reproduction, random death
        death_rate = random.random()
        set_means_and_calc(death_rate,0.0)
        assert_fitnesses(0,0) 

@pytest.mark.skipif(not runRunTests,
        reason="Not running Run class tests.")
class TestRunClass:

    @pytest.mark.parametrize("report_n, verbose",
            [(random.randint(1, 100), True), (random.randint(1, 100), False)])
    def test_init_run(self, conf, report_n, verbose):
        run1 = Run(conf, "", conf.number_of_runs-1, report_n, verbose)
        assert run1.log == ""
        assert run1.n_snap == run1.n_stage == 0
        assert run1.surv_penf == run1.repr_penf == 1.0
        assert run1.resources == conf.res_start
        assert len(run1.genmap) == len(conf.genmap)
        assert not (run1.genmap == conf.genmap).all()
        assert run1.report_n == report_n
        assert run1.verbose == verbose
        assert run1.dieoff == run1.complete == False
        assert run1.n_run == conf.number_of_runs-1
        assert (run1.record.record["genmap"] == run1.genmap).all()
        # Quick test of correct genmap transition from run -> pop -> record;
        # Population and Record initiation tested more thoroughly elsewhere

    def test_update_resources(self, R):
        """Test resource updating between bounds and confirm resources
        cannot go outside them."""
        run1 = copy.copy(R)
        # Constant resources
        run1.conf.res_var = False
        old_res = run1.resources
        run1.update_resources()
        assert run1.resources == old_res
        # Variable resources
        run1.conf.res_var = True
        run1.conf.V, run1.conf.R, run1.conf.res_limit = 2.0, 1000, 5000
        run1.resources, run1.population.N = 5000, 0
        run1.update_resources()
        assert run1.resources == 5000
        run1.resources, run1.population.N = 0, 5000
        run1.update_resources()
        assert run1.resources == 0
        run1.resources, run1.population.N = 1000, 500
        run1.update_resources()
        assert run1.resources == 2000
        run1.resources, run1.population.N = 500, 1000
        run1.update_resources()
        assert run1.resources == 500

    def test_starving(self, R):
        """Test run enters starvation state under correct conditions for
        constant and variable resources."""
        run1 = copy.copy(R)
        # Constant resources
        run1.conf.res_var = False
        run1.resources, run1.population.N = 5000, 4999
        assert not run1.starving()
        run1.resources, run1.population.N = 4999, 5000
        assert run1.starving()
        # Variable resources
        run1.conf.res_var = True
        run1.resources = 1
        assert not run1.starving()
        run1.resources = 0
        assert run1.starving()

    @pytest.mark.parametrize("spen", [True, False])
    @pytest.mark.parametrize("rpen", [True, False])
    def test_update_starvation_factors(self, R, spen, rpen):
        """Test that starvation factors update correctly under various
        conditions."""
        run1 = copy.copy(R)
        run1.conf.surv_pen, run1.conf.repr_pen = spen, rpen
        # Expected changes
        ec_s = run1.conf.death_inc if spen else 1.0
        ec_r = run1.conf.repr_dec  if rpen else 1.0
        # 1: Under non-starvation, factors stay at 1.0
        run1.conf.res_var, run1.resources = True, 1
        run1.update_starvation_factors()
        assert run1.surv_penf == run1.repr_penf == 1.0
        # 2: Under starvation, factors increase
        run1.resources = 0
        run1.update_starvation_factors()
        assert run1.surv_penf == ec_s
        assert run1.repr_penf == ec_r
        # 3: Successive starvation compounds factors exponentially
        run1.update_starvation_factors()
        assert run1.surv_penf == ec_s**2
        assert run1.repr_penf == ec_r**2
        # 4: After starvation ends factors reset to 1.0
        run1.resources = 1
        run1.update_starvation_factors()
        assert run1.surv_penf == 1.0
        assert run1.repr_penf == 1.0

    def test_execute_stage_functionality(self, run):
        """Test functional operations of test_execute_stage, ignoring 
        status reporting."""
        # Normal
        run1 = copy.copy(run)
        run1.population = run1.population.toPop()
        run1.execute_stage()
        assert run1.n_stage == run.n_stage + 1
        assert run1.n_snap == run.n_snap + 1
        assert (run1.dieoff == (run1.population.N == 0))
        assert run1.complete == run1.dieoff
        # Last stage
        run2 = copy.copy(run)
        run2.n_stage = run2.conf.number_of_stages -1
        run2.population = run2.population.toPop()
        run2.execute_stage()
        assert run2.n_stage == run.conf.number_of_stages
        assert (run2.dieoff == (run2.population.N == 0))
        assert run2.complete
        # Dead
        run3 = copy.copy(run)
        run3.population = run3.population.toPop()
        run3.population.N = 0
        run3.population.ages = np.array([])
        run3.population.genomes = np.array([[],[]])
        run3.execute_stage()
        assert run3.n_stage == run.n_stage + 1
        assert run3.n_snap == run.n_snap
        assert run3.dieoff and run3.complete

    @pytest.mark.parametrize("crisis_p,crisis_sv",\
            [(0.0,1.0),(1.0,1.0),(1.0,0.5)])
    def test_execute_stage_degen(self,run,crisis_p,crisis_sv):
        """Test execute_stage operates correctly when there is 0 probability
        of birth, death or crisis death."""
        run1 = copy.copy(run)
        z = np.zeros(2*run1.conf.n_base + 1)
        # update_agestats, update_invstats use record.record
        run1.record.record["d_range"] = np.copy(z)
        run1.record.record["r_range"] = np.copy(z)
        # growth, death use run.conf
        run1.conf.d_range, run1.conf.r_range = np.copy(z),np.copy(z)
        run1.conf.crisis_p,run1.conf.crisis_sv = crisis_p, crisis_sv
        # Other setup
        run1.population.genomes = np.ones(run1.population.genomes.shape).astype(int)
        run1.conf.number_of_stages = 1
        run1.conf.snapshot_stages = [0]#!
        run1.conf.res_var = False
        run1.resources = run1.population.N
        run1.population = run1.population.toPop()
        # Test masks
        r = run1.record.record
        old_N = run1.population.N
        mask = np.zeros(run1.population.maxls)
        ad_mask = np.copy(mask) # Age distribution
        ad_mask[run1.population.maturity] = 1
        density_mask = np.append(np.zeros(2*run1.conf.n_base),[1])
        n1_mask = np.ones(run1.population.chrlen)
        n1_std_mask = np.zeros(run1.population.chrlen)
        s1_mask = np.zeros(run1.record.record["s1"][0].shape)
        # Execute
        run1.execute_stage()
        # record
        def assert_allequal(keys,test):
            for x in keys:
                if isinstance(r[x][0],np.ndarray): 
                    assert np.array_equal(r[x][0],test)
                else: assert r[x][0] == test
        assert_allequal(["surv_penf"], run.surv_penf)
        assert_allequal(["repr_penf"], run.repr_penf)
        assert_allequal(["population_size", "resources"], old_N)
        assert_allequal(["age_distribution"], ad_mask)
        assert_allequal(["age_wise_fitness_product","death_mean","death_sd",
            "age_wise_fitness_contribution","junk_age_wise_fitness_product",
            "junk_age_wise_fitness_contribution","repr_mean","repr_sd"], mask)
        assert_allequal(["density_surv","density_repr"], density_mask)
        assert_allequal(["entropy","junk_death","junk_repr","junk_fitness",
            "fitness"], 0)
        assert_allequal(["n1"], n1_mask)
        assert_allequal(["n1_std"], n1_std_mask)
        assert_allequal(["s1"], s1_mask)
        # population
        assert np.all(run1.population.ages == run1.population.maturity+1)
        assert run1.resources == old_N
        assert run1.surv_penf == run.surv_penf
        assert run1.repr_penf == run.repr_penf
        assert abs(run1.population.N - old_N*crisis_sv) <= 1
        # run status
        assert run1.dieoff == False
        assert run1.n_stage == 1
        assert run1.complete == True

    def test_logprint_run(self, R, ran_str):
        """Test logging (and especially newline) functionality."""
        R2 = copy.deepcopy(R)
        R2.log = ""
        R2.conf.number_of_runs = 1
        R2.conf.number_of_stages = 1
        R2.n_run = 0
        R2.n_stage = 0
        R2.logprint(ran_str)
        assert R2.log == "RUN 0 | STAGE 0 | {0}\n".format(ran_str)
        R2.log = ""
        R2.conf.number_of_runs = 101
        R2.conf.number_of_stages = 101
        R2.logprint(ran_str)
        assert R2.log == "RUN   0 | STAGE   0 | {0}\n".format(ran_str)

@pytest.mark.skipif(not runSimulationTests,
        reason="Not running Simulation class tests.")
class TestSimulationClass:

    @pytest.mark.parametrize("seed,report_n,verbose",\
            [("",1,False), ("",10,False), ("",100,True), 
            ("run_sim",1,True)])
    def test_init_sim(self, S, run_sim, seed, report_n, verbose):
        if seed == "run_sim": seed = run_sim
        S1 = copy.deepcopy(S)
        T = Simulation("config_test", seed, -1, report_n, verbose)
        if seed == "":
            assert T.startpop == [""]
        else: 
            S1.get_startpop(seed, -1)
            s = S1.startpop
            for n in xrange(len(T.startpop)):
                assert np.array_equal(T.startpop[n].genomes, s[n].genomes)
                assert np.array_equal(T.startpop[n].ages, s[n].ages)
                assert np.array_equal(T.startpop[n].chrlen, s[n].chrlen)
                assert np.array_equal(T.startpop[n].nbase, s[n].nbase)
                assert np.array_equal(T.startpop[n].genmap, s[n].genmap)
        assert T.report_n == report_n
        assert T.verbose == verbose
        assert len(T.runs) == T.conf.number_of_runs
        for n in xrange(T.conf.number_of_runs):
            r = T.runs[n]
            assert r.report_n == T.report_n
            assert r.verbose == T.verbose
            if seed == "":
                for k in set(r.conf.__dict__.keys()):
                    print k
                    rck = r.conf.__dict__[k]
                    if k == "genmap": rck = np.sort(rck)
                    test = rck == T.conf.__dict__[k]
                    if isinstance(rck, np.ndarray): test = np.all(test)
                    assert test
            if seed != "":
                s = T.startpop[0] if len(T.startpop) == 1 else T.startpop[n]
                assert np.array_equal(r.population.genomes, s.genomes)
                assert np.array_equal(r.population.ages, s.ages)
                assert r.population.chrlen == s.chrlen
                assert r.population.nbase == s.nbase
                assert np.array_equal(r.population.genmap, s.genmap)

    def test_execute(self, S):
        """Quickly test that execute runs execute_run for every run."""
        S1 = copy.deepcopy(S)
        S1.execute()
        for r in S1.runs:
            assert r.complete

    def test_get_conf_bad(self, S, ran_str):
        """Verify that fn.get_conf throws an error when the target file
        does not exist."""
        S1 = copy.deepcopy(S)
        with pytest.raises(IOError) as e_info: S1.get_conf(ran_str)

    def test_get_conf_good(self, S):
        """Test that get_conf on the config template file returns a valid
        object of the expected composition."""
        S1 = copy.deepcopy(S)
        S1.get_conf("config_test")
        c = S1.conf
        def assert_alltype(keys,typ):
            for x in keys:
                assert isinstance(c.__dict__[x], typ)
        assert_alltype(["number_of_runs", "number_of_stages",
            "number_of_snapshots", "res_start", "R", "res_limit",
            "start_pop", "max_ls", "maturity", "n_base",
            "death_inc", "repr_dec", "window_size"], int)
        assert_alltype(["crisis_sv", "V", "r_rate", "m_rate", "m_ratio",
            "crisis_p"], float)
        assert_alltype(["sexual", "res_var", "age_random", "surv_pen",
                "repr_pen"], bool)
        assert_alltype(["death_bound", "repr_bound", "crisis_stages"],
                    list)

    def test_get_startpop_good(self, S, run_sim):
        """Test that a blank seed returns a list containing a blank string and i
        a valid seed returns a list of populations of the correct size."""
        S1 = copy.deepcopy(S)
        S1.get_startpop("")
        assert S1.startpop == [""]
        px = run_sim
        S1.get_startpop("sample_output.sim", 0)
        assert len(S1.startpop) == 1
        assert S1.startpop[0].genomes.shape==px.runs[0].population.genomes.shape
        S1.get_startpop("sample_output.sim", -1)
        assert len(S1.startpop) == len(px.runs)
        for n in range(len(px.runs)):
            assert S1.startpop[n].genomes.shape == \
                    px.runs[n].population.genomes.shape
        S1.get_startpop(px, 0)
        assert len(S1.startpop) == 1
        assert S1.startpop[0].genomes.shape==px.runs[0].population.genomes.shape


    def test_get_startpop_bad(self, S, ran_str):
        """Verify that fn.get_startpop throws an error when the target
        file does not exist."""
        S1 = copy.deepcopy(S)
        with pytest.raises(IOError) as e_info: S1.get_startpop(ran_str)

    def test_finalise(self, S, run_sim):
        """Test that the simulation correctly creates output files."""
        S1 = copy.deepcopy(S)
        S1.log = ""
        S1.runs = run_sim.runs
        assert not hasattr(S1,"avg_record")
        print [(r.complete, r.dieoff) for r in S1.runs]
        S1.finalise("x_output", "x_log")
        assert isinstance(S1.avg_record, Record)
        for r in S1.runs:
            assert isinstance(r.population, Outpop)
        assert os.stat("x_output.sim").st_size > 0
        assert os.stat("x_output.rec").st_size > 0
        assert os.stat("x_log.txt").st_size > 0
        os.remove("x_output.sim")
        os.remove("x_output.rec")
        os.remove("x_log.txt")

    def test_average_records(self, S, run_sim):
        S1 = copy.deepcopy(S)
        S1.runs = run_sim.runs
        S1.average_records()
        def comp(a, b): assert np.all(np.isclose(a, b))
        def test_keys(n,m): # n = number of failed runs, m = prev_failed per run
            test,ref = S1.avg_record.record,[r.record.record for r in S1.runs[n:]]
            excl = ["prev_failed", "percent_dieoff", "n_runs", "n_successes"]
            avg_keys,l = set(ref[0].keys()) - set(excl), len(S1.runs)
            print sorted(test.keys())
            print sorted(ref[0].keys())
            print sorted(list(avg_keys))
            for key in avg_keys:
                comp(test[key], np.mean([r[key] for r in ref],0) )
                comp(test[key+"_SD"], np.std([r[key] for r in ref],0) )
            assert test["prev_failed"] == l*m
            assert np.isclose(test["percent_dieoff"],100*(l*m+n)/(l*m+l))
            assert test["n_runs"] == l
            assert test["n_successes"] == l-n
        S1.average_records()
        test_keys(0,0)
        S1.runs[0].dieoff = True
        S1.average_records()
        test_keys(1,0)
        for r in S1.runs:
            r.record.record["prev_failed"] = 1
        S1.average_records()
        test_keys(1,1)

    def test_logprint_sim(self, S, ran_str):
        """Test logging (and especially newline) functionality."""
        S1 = copy.deepcopy(S)
        S1.log = ""
        S1.logprint(ran_str)
        assert S1.log == ran_str + "\n"

def test_post_cleanup():
    """Kill tempfiles made for test. Not really a test at all."""
    os.remove("sample_output.sim")
    os.remove("sample_output.rec")
    os.remove("sample_log.txt")
