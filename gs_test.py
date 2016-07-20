# TODO: 
# - test_execute_run (TestFunctions)
# - Write test_execute for Run package
# - Test repr_penf against surv_penf; bug from 4-Jul-2016
# - Remove redundant tests
# - Speed up tests involving get_startpop (currently very slow)
# - Add tests for Config class

import pyximport; pyximport.install()
from gs_core import Simulation, Run, Outpop, Population, Record, Config # Classes
from gs_core import chance, get_runtime, execute_run # Functions
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
def record(request,pop1,conf):
    """Create a record from pop1 as defined in configuration file."""
    spaces = 2 * pop1.nbase + 1
    return Record(pop1, conf.snapshot_stages, conf.number_of_stages,
            np.linspace(1,0,spaces), np.linspace(0,1,spaces),
            conf.window_size)

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
@pytest.fixture()
def S(request):
    """Create an unmodified, unseeded simulation object for procedure testing."""
    return Simulation("config_test", "", -1, 100, False)
@pytest.fixture()
def R(request, S):
    """Create an unmodified, unseeded run object for procedure testing."""
    return Run(S.conf, "", 0, 100, False)

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
    os.rename("log.txt", "sample_log.txt")

#########################
### 1: FREE FUNCTIONS ###
#########################

# -------------------------
# RANDOM NUMBER GENERATION
# -------------------------
@pytest.mark.skipif(not runFunctionConfigTests,
        reason="Not running function/Config tests.")
class TestFunctionsConfig:

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
        assert (c.d_range == np.linspace(c.death_bound[1], 
            c.death_bound[0],2*c.n_base+1)).all()
        assert (c.r_range == np.linspace(c.repr_bound[0], 
            c.repr_bound[1],2*c.n_base+1)).all()
        assert len(c.snapshot_stages) == c.number_of_snapshots if \
                type(nsnap) is int else int(nsnap * c.number_of_stages)
        assert np.all(c.snapshot_stages == np.around(np.linspace(
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
        assert abs(pop.N - p*spop.N)/spop.N < precision

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
class TestRecordClass:
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
        assert run1.surv_penf == run1.repr_penf == 1.0

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
        assert run1.record.record["population_size"][0] == old_N
        assert run1.record.record["surv_penf"][0] == run.surv_penf
        assert run1.record.record["repr_penf"][0] == run.repr_penf
        assert run1.record.record["resources"][0] == old_N
        assert np.all(run1.record.record["age_distribution"][0] == ad_mask)
        assert np.all(run1.record.record["death_mean"][0] == mask)
        assert np.all(run1.record.record["death_sd"][0] == mask)
        assert np.all(run1.record.record["repr_mean"][0] == mask)
        assert np.all(run1.record.record["repr_sd"][0] == mask)
        assert np.all(run1.record.record["fitness"][0] == mask)
        assert np.all(run1.record.record["density_surv"][0] == density_mask)
        assert np.all(run1.record.record["density_repr"][0] == density_mask)
        assert np.all(run1.record.record["n1"][0] == n1_mask)
        assert np.all(run1.record.record["n1_std"][0] == n1_std_mask)
        assert np.all(run1.record.record["s1"][0] == s1_mask)
        assert run1.record.record["entropy"][0] == 0
        assert run1.record.record["junk_death"][0] == 0
        assert run1.record.record["junk_repr"][0] == 0
        assert run1.record.record["junk_fitness"][0] == 0
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
        R.log = ""
        R.conf.number_of_runs = 1
        R.conf.number_of_stages = 1
        R.n_run = 0
        R.n_stage = 0
        R.logprint(ran_str)
        assert R.log == "RUN 0 | STAGE 0 | {0}\n".format(ran_str)
        R.log = ""
        R.conf.number_of_runs = 101
        R.conf.number_of_stages = 101
        R.logprint(ran_str)
        assert R.log == "RUN   0 | STAGE   0 | {0}\n".format(ran_str)

@pytest.mark.skipif(not runSimulationTests,
        reason="Not running Simulation class tests.")
class TestSimulationClass:

    @pytest.mark.parametrize("seed,report_n,verbose",\
            [("",1,False), ("",10,False), ("",100,True), 
            ("sample_output",1,True)])
    def test_init_sim(self, S, seed, report_n, verbose):
        T = Simulation("config_test", seed, -1, report_n, verbose)
        if seed == "":
            assert T.startpop == [""]
        else: 
            S.get_startpop(seed, -1)
            s = S.startpop
            for n in xrange(len(T.startpop)):
                assert np.all(T.startpop[n].genomes == s[n].genomes)
                assert np.all(T.startpop[n].ages == s[n].ages)
                assert T.startpop[n].chrlen == s[n].chrlen
                assert T.startpop[n].nbase == s[n].nbase
                assert np.all(T.startpop[n].genmap == s[n].genmap)
        assert T.report_n == report_n
        assert T.verbose == verbose
        assert len(T.runs) == T.conf.number_of_runs
        for n in xrange(T.conf.number_of_runs):
            r = T.runs[n]
            assert r.report_n == T.report_n
            assert r.verbose == T.verbose
            if seed == "":
                for k in r.conf.__dict__.keys():
                    test = r.conf.__dict__[k] == T.conf.__dict__[k]
                    if isinstance(r.conf.__dict__[k], np.ndarray):
                        assert np.all(test)
                    else:
                        assert test
            if seed != "":
                s = T.startpop[0] if len(T.startpop) == 1 else T.startpop[n]
                assert np.all(r.population.genomes == s.genomes)
                assert np.all(r.population.ages == s.ages)
                assert r.population.chrlen == s.chrlen
                assert r.population.nbase == s.nbase
                assert np.all(r.population.genmap == s.genmap)

    @pytest.mark.xfail(reason="unwritten")
    def test_execute_run(self):
        assert False

    def test_execute(self, S):
        """Quickly test that execute runs execute_run for every run."""
        S.execute()
        for r in S.runs:
            assert r.complete

    def test_get_conf_bad(self, S, ran_str):
        """Verify that fn.get_conf throws an error when the target file
        does not exist."""
        with pytest.raises(IOError) as e_info: S.get_conf(ran_str)

    def test_get_conf_good(self, S):
        """Test that get_conf on the config template file returns a valid
        object of the expected composition."""
        S.get_conf("config_test")
        c = S.conf
        def assert_alltype(keys,typ):
            """Test whether all listed config items are of the
            specified type."""
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

    def test_get_startpop_good(self, S):
        """Test that a blank seed returns a list containing a blank string and i
        a valid seed returns a list of populations of the correct size."""
        S.get_startpop("")
        assert S.startpop == [""]
        try:
            f = open("sample_output.sim", "rb")
            px = cPickle.load(f)
        finally:
            f.close()
        S.get_startpop("sample_output.sim", 0)
        assert len(S.startpop) == 1
        assert S.startpop[0].genomes.shape==px.runs[0].population.genomes.shape
        S.get_startpop("sample_output.sim", -1)
        assert len(S.startpop) == len(px.runs)
        for n in range(len(px.runs)):
            assert S.startpop[n].genomes.shape == \
                    px.runs[n].population.genomes.shape

    def test_get_startpop_bad(self, S, ran_str):
        """Verify that fn.get_startpop throws an error when the target
        file does not exist."""
        with pytest.raises(IOError) as e_info: S.get_startpop(ran_str)

    def test_finalise(self, S):
        """Test that the simulation correctly creates output files."""
        S.log = ""
        #S.startpop = "test"
        S.finalise("x_output", "x_log")
        #assert S.startpop == "" # Not currently deleting startpop
        for r in S.runs:
            assert isinstance(r.population, Outpop)
        assert os.stat("x_output.sim").st_size > 0
        assert os.stat("x_log.txt").st_size > 0
        os.remove("x_output.sim")
        os.remove("x_log.txt")

    def test_logprint_sim(self, S, ran_str):
        """Test logging (and especially newline) functionality."""
        S.log = ""
        S.logprint(ran_str)
        assert S.log == ran_str + "\n"

def test_post_cleanup():
    """Kill tempfiles made for test. Not really a test at all."""
    os.remove("sample_output.sim")
    os.remove("sample_log.txt")
