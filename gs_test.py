# TODO: 
# - Fix imports
# - Move all tests into modules
# - Fix fixtures

from gs_classes import *
from gs_functions import *
import pytest, random, string, subprocess, math

# Begin by running a dummy simulation and saving the output
scriptdir = os.path.split(os.path.realpath(__file__))[0]
os.chdir(scriptdir)
subprocess.call(["python", "genome_simulation.py", "."])
os.rename("run_1_pop.txt", "sample_pop.txt")
os.rename("run_1_rec.txt", "sample_rec.txt")

####################
### 0: FIXTURES  ###
####################

@pytest.fixture(scope="module")
def ran_str(request):
    """Generate a random lowercase ascii string."""
    return \
        ''.join(random.choice(string.ascii_lowercase) for _ in range(50))
@pytest.fixture(scope="session")
def conf(request):
    """Create a default configuration object."""
    return get_conf("config_test")
@pytest.fixture(scope="session")
def spop(request):
    """Create a sample population from the default configuration."""
    return get_startpop("sample_pop")
@pytest.fixture
def parents(request, conf):
    """Returns population of two sexual adults."""
    params = conf.params.copy()
    params["sexual"] = True
    params["age_random"] = False
    params["start_pop"] = 2
    return Population(params, conf.genmap)

#########################
### 1: FREE FUNCTIONS ###
#########################

class TestConfig:
    """Test that the initial simulation configuration is performed
    correctly."""

    def test_get_dir_good(self):
        """Verify that get_dir functions correctly when given (a) the
        current directory, (b) the parent directory, (c) the root
        directory."""
        old_dir = os.getcwd()
        old_path = sys.path[:]
        get_dir(old_dir)
        same_dir = os.getcwd()
        same_path = sys.path[:]
        test = (same_dir == old_dir and same_path == old_path)
        if old_dir != "/":
            get_dir("..")
            par_dir = os.getcwd()
            par_path = sys.path[:]
            exp_path = [par_dir] + [x for x in old_path if x != old_dir]
            test = (test and par_dir == os.path.split(old_dir)[0])
            test = (test and par_path == exp_path)
            if par_dir != "/":
                get_dir("/")
                root_dir = os.getcwd()
                root_path = sys.path[:]
                exp_path = ["/"] + [x for x in par_path if x != par_dir]
                test = (test and root_dir=="/" and root_path==exp_path)
            get_dir(old_dir)
        assert test

    def test_get_dir_bad(self, ran_str):
        """Verify that get_dir throws an error when the target directory 
        does not exist."""
        with pytest.raises(SystemExit) as e_info: get_dir(ran_str)

    def test_get_conf_good(self, conf):
        """Test that get_conf on the config template file returns a valid
        object of the expected composition."""
        def alltype(keys,typ):
            """Test whether all listed config items are of the 
            specified type."""
            return np.all([type(conf.__dict__[x]) is typ for x in keys])
        assert alltype(["number_of_runs", "number_of_stages",
            "number_of_snapshots", "res_start", "R", "res_limit",
            "start_pop", "max_ls", "maturity", "n_base",
            "death_inc", "repr_dec", "window_size", "chr_len"], int) and\
            alltype(["crisis_sv", "V", "r_rate", "m_rate", "m_ratio"],
                    float) and\
            alltype(["sexual", "res_var", "age_random", "surv_pen",
                "repr_pen"], bool) and\
            alltype(["death_bound", "repr_bound", "crisis_stages"],
                    list) and\
            alltype(["g_dist", "params"], dict) and\
            alltype(["genmap", "d_range", "r_range", 
                "snapshot_stages"], np.ndarray)

    def test_get_conf_bad(self, ran_str):
        """Verify that get_dir throws an error when the target file does
        not exist."""
        with pytest.raises(IOError) as e_info: get_conf(ran_str)

    def test_get_startpop_good(self, conf, spop):
        """Test that a blank seed returns a blank string and a valid seed
        returns a population array of the correct size."""
        assert get_startpop("") == "" and\
                spop.genomes.shape == (spop.N, 2*spop.chrlen)

    def test_get_startpop_bad(self, ran_str):
        """Verify that get_dir throws an error when the target file does
        not exist."""
        with pytest.raises(IOError) as e_info: get_startpop(ran_str)

# -------------------------
# RANDOM NUMBER GENERATION
# -------------------------

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
        assert c.shape == shape and c.dtype == "bool" and \
                abs(p-np.mean(c)) < precision

class TestGenomeArray:
    """Test that new genome arrays are generated correctly."""
    genmap_simple = np.append(np.arange(25), 
            np.append(np.arange(24)+100, 201))
    genmap_shuffled = np.copy(genmap_simple)
    random.shuffle(genmap_shuffled)

    @pytest.mark.parametrize("g_dist", [
        {"s":random.random(), "r":random.random(), "n":random.random()},
        {"s":0.5, "r":0.5, "n":0.5},
        {"s":0.1, "r":0.9, "n":0.4}])
    @pytest.mark.parametrize("genmap", [genmap_simple, genmap_shuffled])
    def test_make_genome_array(self, genmap, g_dist):
        """Test that genome array is of the correct size and that
        the loci are distributed correctly."""
        precision = 0.05
        loci = {
            "s":np.nonzero(genmap<100)[0],
            "r":np.nonzero(np.logical_and(genmap>=100,genmap<200))[0],
            "n":np.nonzero(genmap>=200)[0]
            }
        n = 1000
        chr_len = 500
        ga = make_genome_array(n, chr_len, genmap, 10, g_dist)
        test = ga.shape == (n, 2*chr_len)
        condensed = np.mean(ga, 0)
        condensed = np.array([np.mean(condensed[x*10:(x*10+9)]) \
                for x in range(chr_len/5)])
        for k in loci.keys():
            pos = np.array([range(10) + x for x in loci[k]*10])
            pos = np.append(pos, pos + chr_len)
            tstat = abs(np.mean(ga[:,pos])-g_dist[k])
            test = test and tstat < precision
        assert test

class TestUpdateResources:
    """Confirm that resources are updated correctly in the variable-
    resources condition."""

    def test_update_resources_bounded(self):
        """Confirm that resources cannot exceed upper bound or go below
        zero."""
        assert update_resources(5000, 0, 1000, 2, 5000) == 5000 and\
                update_resources(0, 5000, 1000, 2, 5000) == 0
    
    def test_update_resources_unbounded(self):
        """Test resource updating between bounds."""
        assert update_resources(1000, 500, 1000, 2, 5000) == 2000 and\
                update_resources(500, 1000, 1000, 2, 5000) == 500

###########################
### 2: POPULATION CLASS ###
###########################
class TestPopInit:
    """Test initialisation of a population object."""
    def test_init_population(self, conf):
        """Test that population parameters are correct for random and
        nonrandom ages."""
        precision = 1
        conf.params["start_pop"] = 2000
        conf.params["age_random"] = False
        pop1 = Population(conf.params, conf.genmap)
        conf.params["age_random"] = True
        pop2 = Population(conf.params, conf.genmap)
        print np.mean(pop2.ages)
        print abs(np.mean(pop2.ages)-pop2.maxls/2)
        assert \
            pop1.sex == pop2.sex == conf.params["sexual"] and \
            pop1.chrlen == pop2.chrlen == conf.params["chr_len"] and \
            pop1.nbase == pop2.nbase == conf.params["n_base"] and \
            pop1.maxls == pop2.maxls == conf.params["max_ls"] and \
            pop1.maturity==pop2.maturity == conf.params["maturity"] and \
            pop1.N == pop2.N == conf.params["start_pop"] and\
            (pop1.index==np.arange(conf.params["start_pop"])).all() and\
            (pop2.index==np.arange(conf.params["start_pop"])).all() and\
            (pop1.genmap == conf.genmap).all() and\
            (pop2.genmap == conf.genmap).all() and\
            (pop1.ages == pop1.maturity).all() and\
            not (pop2.ages == pop2.maturity).all() and\
            abs(np.mean(pop2.ages)-pop2.maxls/2) < precision
            # make_genome_array is tested separately

class TestMinorMethods:
    """Test whether population minor methods (not death or reproduction)
    work correctly."""
    def test_shuffle(self, spop):
        """Test if all ages, therefore individuals, present before the 
        shuffle are also present after it."""
        spop2 = spop.clone() # clone tested separately
        spop2.shuffle()
        is_shuffled = \
                not (spop.genomes == spop2.genomes).all()
        spop.ages.sort()
        spop2.ages.sort()
        assert is_shuffled and \
                (spop.ages == spop2.ages).all()

    def test_clone(self, spop):
        """Test if cloned population is identical to parent population, 
        by comparing params, ages, genomes."""
        spop2 = spop.clone()
        assert \
        spop.params() == spop2.params() and \
        (spop.genmap == spop2.genmap).all() and \
        (spop.ages == spop2.ages).all() and \
        (spop.genomes == spop2.genomes).all()

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
        assert len(p) >= 5 and p["sexual"] == spop.sex and \
                p["chr_len"] == spop.chrlen and\
                p["n_base"] == spop.nbase and \
                p["max_ls"] == spop.maxls and\
                p["maturity"] == spop.maturity

    def test_addto(self, spop):
        """Test if a population is successfully appended to the receiver
        population, which remains unchanged, by appending a population to
        itself."""
        pop1 = spop.clone()
        pop2 = spop.clone()
        pop2.addto(pop1)
        assert (pop2.ages == np.tile(pop1.ages,2)).all() and \
                (pop2.genomes == np.tile(pop1.genomes,(2,1))).all()

class TestDeathCrisis:
    """Test functioning of get_subpop, death and crisis methods."""

    @pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
    @pytest.mark.parametrize("min_age,offset",[(0,0),(16,100)])
    def test_get_subpop(self, spop, p, min_age, offset):
        precision = 0.1
        """Test if the probability of passing is close to that indicated
        by the genome (when all loci have the same distribution)."""
        pop = spop.clone()
        pop.genomes = chance(p, pop.genomes.shape).astype(int)
        pop2 = pop.get_subpop(min_age, pop.maxls, offset,
                np.linspace(0,1,21))
        assert abs(pop2.N/float(pop.N) - p)*(1-min_age/pop.maxls) < \
                precision

    @pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
    @pytest.mark.parametrize("x", [1.0, 3.0, 9.0])
    def test_death(self, spop, p, x):
        """Test if self.death() correctly inverts death probabilities
        and incorporates starvation factor to get survivor probabilities
        and survivor array."""
        precision = 0.1
        pop = spop.clone()
        b = pop.nbase
        surv_loci = np.nonzero(spop.genmap<100)[0]
        surv_pos = np.array([range(b) + x for x in surv_loci*b])
        surv_pos = np.append(surv_pos, surv_pos + pop.chrlen)
        pop.genomes[:, surv_pos] =\
                chance(p, pop.genomes[:, surv_pos].shape).astype(int)
        # (specifically modify survival loci only)
        pop2 = pop.clone()
        pop2.death(np.linspace(1,0,21), x, False)
        pmod = max(0, min(1, (1-x*(1-p))))
        assert abs(pop2.N/float(pop.N) - pmod) < precision
        # TODO: Increase precision after simulation is more optimised

    @pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
    def test_crisis(self, spop,p):
        """Test whether extrinsic death crisis removes the expected
        fraction of the population."""
        precision = 0.1
        pop = spop.clone()
        pop.crisis(p, "0")
        assert abs(pop.N - p*spop.N) < precision

class TestReproduction:
    """Test methods associated with population reproduction (sexual
    and asexual."""
    def test_recombine_none(self, spop):
        """Test if genome stays same if recombination chance is zero."""
        pop = spop.clone()
        pop._Population__recombine(0)
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
        pop = Population(conf.params, conf.genmap)
        zz = recombine_zig_zag(pop)
        pop._Population__recombine(1)
        assert (pop.genomes == zz).all()

    def test_assortment(self, parents):
        """Test if assortment of two adults is done properly by 
        comparing the function result with one of the expected 
        results.""" 
        parent1 = np.copy(parents.genomes[0])
        parent2 = np.copy(parents.genomes[1])
        c = parents.chrlen
        children = parents._Population__assortment().genomes
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
        spop._Population__mutate(mrate,1)
        assert abs((1-np.mean(genomes == spop.genomes))-mrate) < 0.01

    @pytest.mark.parametrize("mratio", [0, 0.1, 0.5, 1])
    def test_mutate_biased(self, spop, mratio):
        """Test that the bias between positive and negative mutations is
        implemented correctly."""
        t = 0.01 # Tolerance
        mrate = 0.5 
        g0 = np.copy(spop.genomes)
        spop._Population__mutate(mrate,mratio)
        g1 = spop.genomes
        is1 = (g0==1)
        is0 = np.logical_not(is1)
        assert abs((1-np.mean(g0[is1] == g1[is1]))-mrate) < t and\
            abs((1-np.mean(g0[is0] == g1[is0]))-mrate*mratio) < t 

    @pytest.mark.parametrize("sexvar",[True, False])
    @pytest.mark.parametrize("m",[0.0, 0.3, 0.8, 1.0])
    def test_growth(self,conf,sexvar,m):
        """Test number of children produced for all-adult population
        for sexual and asexual conditions."""
        # Make and grow population
        precision = 0.05
        n = 1000
        params = conf.params.copy()
        params["sexual"] = sexvar
        params["age_random"] = False
        params["start_pop"] = n
        pop = Population(params,conf.genmap)
        pop.genomes = chance(m, pop.genomes.shape).astype(int)
        pop.growth(np.linspace(0,1,21),1,0,0,1,False)
        # Calculate proportional observed and expected growth
        x = 2 if sexvar else 1
        obs_growth = (pop.N - n)/float(n)
        exp_growth = m/x
        assert abs(exp_growth-obs_growth) < precision

#######################
### 3: RECORD CLASS ###
#######################

@pytest.fixture
def population1(request, conf):
    """Create population with genomes filled with ones."""
    conf.params["start_pop"] = 500
    conf.params["age_random"] = False
    conf.params["sexual"] = False
    pop = Population(conf.params, conf.genmap)
    pop.genomes = np.ones(pop.genomes.shape).astype(int)
    return pop

### RECORD
def test_init_record(population1, conf):
    r = Record(population1, np.array([10]), 100, np.linspace(0,1,21), 
            np.linspace(0,1,21),100).record
    def sameshape(keys,ref):
        """Test whether all listed record arrays have identical shape."""
        return np.all([r[x].shape == ref for x in keys])
    assert len(r.keys()) == 27 and\
            (r["genmap"] == population1.genmap).all() and\
            (r["chr_len"] == np.array([population1.chrlen])).all() and\
            (r["n_bases"] == np.array([population1.nbase])).all() and\
            (r["max_ls"] == np.array([population1.maxls])).all() and\
            (r["maturity"] == np.array([population1.maturity])).all() and\
            (r["d_range"] == np.linspace(0,1,21)).all() and\
            (r["d_range"] == r["r_range"]).all() and\
            (r["snapshot_stages"] == np.array([11])).all() and\
            sameshape(["death_mean", "death_sd", "repr_mean", "repr_sd",
                "fitness"], (1,population1.maxls)) and\
            sameshape(["density_surv", "density_repr"],
                (1,2*population1.nbase+1)) and\
            sameshape(["entropy","junk_death","junk_repr",
                "junk_fitness"], (1,)) and\
            sameshape(["population_size", "resources", "surv_penf", 
                "repr_penf"], (100,)) and\
            sameshape(["age_distribution"],(100,population1.maxls)) and\
            sameshape(["n1", "n1_std"], (1,population1.chrlen)) and\
            sameshape(["s1"], (1, population1.chrlen - 99))


# not testing quick_update

@pytest.fixture # scope="session"
def record(request,spop,conf):
    """Create a record as defined in configuration file."""
    return Record(spop,conf.snapshot_stages, conf.number_of_stages,
            conf.d_range, conf.r_range, conf.window_size)

def test_update_agestats(record,population1):
    """Test if update_agestats properly calculates agestats for population1
    (genomes filled with ones)."""
    pop = population1.clone()
    record.update_agestats(pop,0)
    r = record.record
    assert \
    np.isclose(r["death_mean"][0], np.tile(r["d_range"][-1],r["max_ls"])).all() and \
    np.isclose(r["death_sd"][0], np.zeros(r["max_ls"])).all() and \
    np.isclose(r["repr_mean"][0], np.append(np.zeros(r["maturity"]),np.tile(r["r_range"][-1],r["max_ls"]-r["maturity"]))).all() and \
    np.isclose(r["repr_sd"][0], np.zeros(r["max_ls"])).all() and \
    r["density_surv"][0][-1] == 1 and \
    r["density_repr"][0][-1] == 1

def test_update_shannon_weaver(record,population1):
    """Test if equals zero when all set members are of same type."""
    assert record.update_shannon_weaver(population1) == -0

#def test_sort_n1(record):
#    """Test if sort_n1 correctly sorts an artificially created genome array."""
#    genmap = record.record["genmap"]
#
#    ix = np.arange(len(genmap))
#    np.random.shuffle(ix)
#    record.record["genmap"] = genmap[ix]
#
#    genome_foo = np.tile(ix.reshape((len(ix),1)),10)
#    genome_foo = genome_foo.reshape((1,len(genome_foo)*10))[0]
#    mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),10)
#    mask = mask.reshape((1,len(mask)*10))[0]
#
#    assert (record.sort_n1(genome_foo) == mask).all()

def test_age_wise_n1(record):
    """Test if ten conecutive array items are correctly averaged."""
    genmap = record.record["genmap"]
    ix = np.arange(len(genmap))
    mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),10)
    mask = mask.reshape((1,len(mask)*10))[0]
    record.record["mask"] = np.array([mask])
    assert (record.age_wise_n1("mask") == ix).all()

def test_update_invstats(record,population1):
    """Test if update_invstats properly calculates genomestats for population1
    (genomes filled with ones)."""
    pop = population1.clone()
    record.update_invstats(pop,0)
    r = record.record
    assert \
    (r["n1"][0] == np.ones(r["chr_len"])).all() and \
    (r["n1_std"][0] == np.zeros(r["chr_len"])).all() and \
    r["entropy"][0] == -0 and \
    np.isclose(r["junk_death"][0], r["d_range"][-1]) and \
    np.isclose(r["junk_repr"][0], r["r_range"][-1])

# not testing final_update

def test_actual_death_rate(record):
    """Test if actual_death_rate returns expected results for artificial data."""
    r = record.record
    maxls = r["max_ls"]
    r["age_distribution"] = np.array([np.tile(4/282.0,maxls),np.tile(2/142.0,maxls),np.tile(1/71.0,maxls)])
    r["population_size"] = np.array([282,142,71])

    assert (record.actual_death_rate()[:,:-1] == np.tile(0.5,maxls-1)).all() and\
            (record.actual_death_rate()[:,-1] == [1]).all()
