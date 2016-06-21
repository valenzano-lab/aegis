# test module for functions in gs_classes

from gs_classes import *
from gs_functions import *
import pytest, random, string, subprocess, math

# skipif marker variable
skipslow = True

# Begin by running a dummy simulation and saving the output
scriptdir = os.path.split(os.path.realpath(__file__))[0]
print scriptdir
os.chdir(scriptdir)
subprocess.call(["python", "genome_simulation.py", "."])
os.rename("run_1_pop.txt", "test_pop.txt")
os.rename("run_1_rec.txt", "test_rec.txt")

#########################
### 1: FREE FUNCTIONS ###
#########################

# --------------
# CONFIGURATION
# --------------

def test_get_dir_good():
    """Verify that get_dir functions correctly when given (a) the current
    directory, (b) the parent directory, (c) the root directory."""
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
            test = (test and root_dir == "/" and root_path == exp_path)
        get_dir(old_dir)
    assert test

def test_get_dir_bad():
    """Verify that get_dir throws an error when the target directory does
    not exist."""
    ranstr = ''.join(random.choice(string.ascii_lowercase) for _ in range(50))
    with pytest.raises(SystemExit) as e_info:
        get_dir(ranstr)

def test_get_conf_good():
    """Test that get_conf on the config template file returns a valid 
    object of the expected composition."""
    p = get_conf("config")
    exp = ['R', 'V', '__builtins__', '__doc__', '__file__', '__name__', 
            '__package__', 'age_random', 'chr_len', 'crisis_stages', 
            'crisis_sv', 'd_range', 'death_bound', 'death_inc', 'g_dist',
            'genmap', 'm_rate', 'm_ratio', 'maturity', 'max_ls', 
            'n_base', 'np', 'number_of_runs', 'number_of_snapshots', 
            'number_of_stages', 'params', 'r_range', 'r_rate', 
            'repr_bound', 'repr_dec', 'repr_pen', 'res_limit', 
            'res_start', 'res_var', 'sexual', 'snapshot_stages', 
            'start_pop', 'surv_pen', 'window_size']
    assert sorted(p.__dict__.keys()) == exp

def test_get_conf_bad():
    """Verify that get_dir throws an error when the target file does
    not exist."""
    ranstr = ''.join(random.choice(string.ascii_lowercase) for _ in range(50))
    with pytest.raises(IOError) as e_info:
        get_conf(ranstr)

def test_get_startpop_good():
    """Test that a blank seed returns a blank string and a valid seed
    returns a population array of the correct file."""
    c = get_conf("config")
    p = get_startpop("test_pop")
    assert get_startpop("") == "" and p.genomes.shape == (p.N, 2*p.chrlen)

def test_get_startpop_bad():
    """Verify that get_dir throws an error when the target file does
    not exist."""
    ranstr = ''.join(random.choice(string.ascii_lowercase) for _ in range(50))
    with pytest.raises(IOError) as e_info:
        get_startpop(ranstr)

# -------------------------
# RANDOM NUMBER GENERATION
# -------------------------

@pytest.mark.parametrize("arg1, arg2", [(0,1), (1,1)])
def test_chance_degenerate(arg1, arg2):
    """Tests wether p=1 returns True/1 and p=0 returns False/0."""
    ans = chance(arg1, arg2).astype(int)
    assert ans == arg1 or (ans > arg1-0.001 and ans < arg1+0.001)

@pytest.mark.skipif(skipslow, reason="Skipping slow tests.")
@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_chance(p):
    """Test that the shape of the output is correct and that the mean
    over many trials is close to the expected value."""
    c = chance(p, (10000,10000))
    s = c.shape
    assert c.shape == (10000,10000) and c.dtype == "bool" and \
            abs(p-np.mean(c)) < 0.001

# ------------------------
# GENOME ARRAY GENERATION
# ------------------------

genmap_simple = np.append(np.arange(25), 
        np.append(np.arange(24)+100, 201))
genmap_shuffled = np.copy(genmap_simple)
random.shuffle(genmap_shuffled)

@pytest.mark.parametrize("g_dist", [
    {"s":random.random(), "r":random.random(), "n":random.random()},
    {"s":0.5, "r":0.5, "n":0.5},
    {"s":0.1, "r":0.9, "n":0.4}])
@pytest.mark.parametrize("genmap", [genmap_simple, genmap_shuffled])
def test_mga(genmap, g_dist):
    """Test that genome array is of the correct size and that 
    the loci are distributed equally for small dummy parameters."""
    loci = {
        "s":np.nonzero(genmap<100)[0],
        "r":np.nonzero(np.logical_and(genmap>=100,genmap<200))[0],
        "n":np.nonzero(genmap>=200)[0]
        }
    n = 5000
    chr_len = 500
    ga = make_genome_array(n, chr_len, genmap, 10, g_dist)
    test = ga.shape == (n, 2*chr_len)
    condensed = np.mean(ga, 0)
    condensed = np.array([np.mean(condensed[x*10:(x*10+9)]) for x in range(chr_len/5)])
    print condensed
    for k in loci.keys():
        pos = np.array([range(10) + x for x in loci[k]*10])
        pos = np.append(pos, pos + chr_len)
        tstat = abs(np.mean(ga[:,pos])-g_dist[k])
        test = test and tstat < 0.01
        print k, tstat
        print genmap[loci[k]]
    assert test

# ------------------
# RESOURCE UPDATING
# ------------------

def test_update_resources_bounded():
    """Confirm that resources cannot exceed upper bound or go below
    zero."""
    assert update_resources(5000, 0, 1000, 2, 5000) == 5000 and\
            update_resources(0, 5000, 1000, 2, 5000) == 0

def test_update_resources_unbounded():
    """Test resource updating between bounds."""
    assert update_resources(1000, 500, 1000, 2, 5000) == 2000 and\
            update_resources(500, 1000, 1000, 2, 5000) == 500

###########################
### 2: POPULATION CLASS ###
###########################

@pytest.fixture
def conf(request):
    return get_conf("config")
def pop(request):
    return get_startpop("test_pop")

start_pop = get_conf("config").params["start_pop"]

@pytest.fixture # scope="session"
def population(request, conf):
    """Create a population as defined in configuration file."""
    conf.params["start_pop"] = 500
    conf.params["age_random"] = False
    conf.params["sexual"] = False
    return Population(conf.params, conf.genmap)

def test__init__(conf):
    """Test if parameters are equal in initialized population and config file."""
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
    pop1.maturity == pop2.maturity == conf.params["maturity"] and \
    pop1.N == pop2.N == conf.params["start_pop"] and\
    (pop1.index == np.arange(conf.params["start_pop"])).all() and\
    (pop2.index == np.arange(conf.params["start_pop"])).all() and\
    (pop1.genmap == conf.genmap).all() and\
    (pop2.genmap == conf.genmap).all() and\
    (pop1.ages == pop1.maturity).all() and\
    not (pop2.ages == pop2.maturity).all() and\
    abs(np.mean(pop2.ages)-pop2.maxls/2) < 1
    # make_genome_array is tested separately

#---------------
# MINOR METHODS
#---------------

def test_shuffle(population):
    """Test if all ages, therefore individuals, present before the shuffle are
    also present after it."""
    population2 = population.clone() # clone tested separately
    population2.shuffle()
    is_shuffled = not (population.genomes == population2.genomes).all()
    population.ages.sort()
    population2.ages.sort()
    assert is_shuffled and (population.ages == population2.ages).all()

def test_clone(population):
    """Test if cloned population is identical to parent population, by
    comparing params, ages, genomes."""
    population2 = population.clone()
    assert \
    population.params() == population2.params() and \
    (population.genmap == population2.genmap).all() and \
    (population.ages == population2.ages).all() and \
    (population.genomes == population2.genomes).all()

def test_increment_ages(population):
    """Test if all ages are incrementd by one."""
    ages1 = np.copy(population.ages)
    population.increment_ages()
    ages2 = population.ages
    assert (ages1+1 == ages2).all()

def test_params(population):
    """Test that params returns (at least) the required information."""
    pop = population
    p = population.params()
    assert len(p) >= 5 and p["sexual"] == pop.sex and \
            p["chr_len"] == pop.chrlen and p["n_base"] == pop.nbase and \
            p["max_ls"] == pop.maxls and p["maturity"] == pop.maturity

def test_addto(population):
    """Test if a population is successfully appended to the receiver population,
    which remains unchanged, by appending a population to itself."""
    pop1 = population.clone()
    pop2 = population.clone()
    pop2.addto(pop1)
    assert (pop2.ages == np.tile(pop1.ages,2)).all() and \
            (pop2.genomes == np.tile(pop1.genomes,(2,1))).all()

# Major methods

@pytest.fixture
def population1(request, conf):
    """Create population with genomes filled with ones."""
    conf.params["start_pop"] = 500
    conf.params["age_random"] = False
    conf.params["sexual"] = False
    pop = Population(conf.params, conf.genmap)
    pop.genomes = np.ones(pop.genomes.shape).astype(int)
    return pop

@pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
@pytest.mark.parametrize("min_age,offset",[(0,0),(16,100)])
def test_get_subpop(conf, p, min_age, offset):
    """Test if the probability of passing is close to that indicated
    by the genome."""
    conf.params["start_pop"] = 500
    conf.params["age_random"] = False
    conf.params["sexual"] = False
    pop = Population(conf.params, conf.genmap)
    pop.genomes = chance(p, pop.genomes.shape).astype(int)
    spop = pop.get_subpop(min_age, pop.maxls, offset,
            np.linspace(0,1,21))
    assert abs(spop.N/float(pop.N) - p)*(1-min_age/pop.maxls) < 0.1
    # TODO: Add test for locus-specificity (e.g. only reprodv loci)

@pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
@pytest.mark.parametrize("x", [1.0, 3.0, 9.0])
def test_death(conf, p, x):
    """Test if self.death() correctly inverts death probabilities
    and incorporates starvation factor to get survivor probabilities
    and survivor array."""
    conf.params["start_pop"] = 500
    conf.params["age_random"] = False
    conf.params["sexual"] = False
    pop = Population(conf.params, conf.genmap)
    pop.genomes = chance(p, pop.genomes.shape).astype(int)
    spop = pop.clone()
    spop.death(np.linspace(1,0,21), x, False)
    pmod = max(0, min(1, (1-x*(1-p))))
    assert abs(spop.N/float(pop.N) - pmod) < 0.1
    # TODO: Increase precision after simulation is more optimised

@pytest.mark.parametrize("p", [0, 0.3, 0.8, 1])
def test_crisis(population,p):
    """Test whether extrinsic death crisis removes the expected
    fraction of the population."""
    pop = population.clone()
    pop.crisis(p, "0")
    assert abs(pop.N - p*population.N) < 0.1

# TODO: Add test that an object is a valid population, use instead
# of ad-hoc (or absent) tests in various places

# ----------------
# PRIVATE METHODS - recombination, assortment, mutation
#-----------------

def test_recombine_none(population):
    """Test if genome stays same if recombination chance is zero."""
    pop = population.clone()
    pop._Population__recombine(0)
    assert (pop.genomes == population.genomes).all()


@pytest.mark.skipif(skipslow, reason="Skipping slow tests.")
def test_recombine_all(population):
    """Test if resulting genomee is equal to recombine_zig_zag, when
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
    pop = population.clone()
    pop._Population__recombine(1)
    assert (pop.genomes == recombine_zig_zag(population)).all()

@pytest.fixture
def parents(conf):
    """Returns population of two adults."""
    params = conf.params.copy()
    params["sexual"] = True
    params["age_random"] = False
    params["start_pop"] = 2
    return Population(params, conf.genmap)

def test_assortment(parents):
    """Test if assortment of two adults is done properly by comparing the
    function result with one of the expected results. Safeguards for pop.sex
    and pop.N are covered in the parents fixture, which test_assortment uses."""
    parent1 = np.copy(parents.genomes[0])
    parent2 = np.copy(parents.genomes[1])
    chrlen = parents.chrlen

    children = parents._Population__assortment().genomes

    assert \
    (children == np.append(parent1[:chrlen], parent2[:chrlen])).all() or \
    (children == np.append(parent2[:chrlen], parent1[:chrlen])).all() or \
    (children == np.append(parent1[:chrlen], parent2[chrlen:])).all() or \
    (children == np.append(parent2[:chrlen], parent1[chrlen:])).all() or \
    (children == np.append(parent1[chrlen:], parent2[:chrlen])).all() or \
    (children == np.append(parent2[chrlen:], parent1[:chrlen])).all() or \
    (children == np.append(parent1[chrlen:], parent2[chrlen:])).all() or \
    (children == np.append(parent2[chrlen:], parent1[chrlen:])).all()

@pytest.mark.parametrize("mrate", [0, 0.3, 0.8, 1]) 
def test_mutate_unbiased(population, mrate):
    """Test that, in the absence of a +/- bias, the appropriate
    proportion of the genome is mutated."""
    genomes = np.copy(population.genomes)
    population._Population__mutate(mrate,1)
    assert abs((1-np.mean(genomes == population.genomes))-mrate) < 0.01

@pytest.mark.parametrize("mratio", [0, 0.1, 0.5, 1])
def test_mutate_biased(population, mratio):
    """Test that the bias between positive and negative mutations is
    implemented correctly."""
    t = 0.01 # Tolerance
    mrate = 0.5 
    g0 = np.copy(population.genomes)
    population._Population__mutate(mrate,mratio)
    g1 = population.genomes
    is1 = (g0==1)
    is0 = np.logical_not(is1)
    assert abs((1-np.mean(g0[is1] == g1[is1]))-mrate) < t and\
        abs((1-np.mean(g0[is0] == g1[is0]))-mrate*mratio) < t 

# -------
# GROWTH

@pytest.mark.parametrize("sexvar",[True, False])
@pytest.mark.parametrize("m",[0.0, 0.3, 0.8, 1.0])
def test_growth(conf,sexvar,m):
    """Test number of children produced for all-adult population
    for sexual and asexual conditions."""
    # Make and grow population
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
    assert abs(exp_growth-obs_growth)<0.05

### RECORD

# not testing init

# not testing quick_update

@pytest.fixture # scope="session"
def record(request,population,conf):
    """Create a record as defined in configuration file."""
    return Record(population,conf.snapshot_stages, conf.number_of_stages,
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

def test_sort_n1(record):
    """Test if sort_n1 correctly sorts an artificially created genome array."""
    genmap = record.record["genmap"]

    ix = np.arange(len(genmap))
    np.random.shuffle(ix)
    record.record["genmap"] = genmap[ix]

    genome_foo = np.tile(ix.reshape((len(ix),1)),10)
    genome_foo = genome_foo.reshape((1,len(genome_foo)*10))[0]
    mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),10)
    mask = mask.reshape((1,len(mask)*10))[0]

    assert (record.sort_n1(genome_foo) == mask).all()

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
