# TODO:
# - Fix imports

from gs_classes import *
from gs_functions import *
import pytest, random, string, subprocess, math, copy
from scipy.misc import comb

# Begin by running a dummy simulation and saving the output
# Also functions as test of output functions
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
@pytest.fixture(scope="session")
def pop1(request, spop):
    """Create population with genomes filled with ones."""
    pop = spop.clone()
    pop.genomes = np.ones(pop.genomes.shape).astype(int)
    return pop
@pytest.fixture
def record(request,pop1,conf):
    """Create a record from pop1 as defined in configuration file."""
    return Record(pop1, conf.snapshot_stages, 100, np.linspace(1,0,21), 
                np.linspace(0,1,21),100)

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
    def test_init_population(self, record, conf):
        """Test that population parameters are correct for random and
        nonrandom ages."""
        precision = 1
        conf.params["start_pop"] = 2000
        conf.params["age_random"] = False
        pop_a = Population(conf.params, conf.genmap)
        conf.params["age_random"] = True
        pop_b = Population(conf.params, conf.genmap)
        print np.mean(pop_b.ages)
        print abs(np.mean(pop_b.ages)-pop_b.maxls/2)
        assert \
            pop_a.sex == pop_b.sex == conf.params["sexual"] and \
            pop_a.chrlen == pop_b.chrlen == conf.params["chr_len"] and \
            pop_a.nbase == pop_b.nbase == conf.params["n_base"] and \
            pop_a.maxls == pop_b.maxls == conf.params["max_ls"] and \
            pop_a.maturity==pop_b.maturity == conf.params["maturity"] and \
            pop_a.N == pop_b.N == conf.params["start_pop"] and\
            (pop_a.index==np.arange(conf.params["start_pop"])).all() and\
            (pop_b.index==np.arange(conf.params["start_pop"])).all() and\
            (pop_a.genmap == conf.genmap).all() and\
            (pop_b.genmap == conf.genmap).all() and\
            (pop_a.ages == pop_a.maturity).all() and\
            not (pop_b.ages == pop_b.maturity).all() and\
            abs(np.mean(pop_b.ages)-pop_b.maxls/2) < precision
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
        pop_a = spop.clone()
        pop_b = spop.clone()
        pop_b.addto(pop_a)
        assert (pop_b.ages == np.tile(pop_a.ages,2)).all() and \
                (pop_b.genomes == np.tile(pop_a.genomes,(2,1))).all()

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

### RECORD
class TestRecordInit:
    """Test the initiation of a new Record object."""
    def test_init_record(self, record, conf, pop1):
        r = record.record
        n = len(r["snapshot_stages"])
        def sameshape(keys,ref):
            """Test whether all listed record arrays have identical 
            shape."""
            return np.all([r[x].shape == ref for x in keys])
        assert (r["genmap"] == pop1.genmap).all()
        assert (r["chr_len"] == np.array([pop1.chrlen])).all()
        assert (r["n_bases"] == np.array([pop1.nbase])).all()
        assert (r["max_ls"] == np.array([pop1.maxls])).all()
        assert (r["maturity"] == np.array([pop1.maturity])).all()
        assert (r["d_range"] == np.linspace(1,0,21)).all()
        assert (r["r_range"] == np.linspace(0,1,21)).all()
        assert (r["snapshot_stages"] == conf.snapshot_stages + 1).all()
        assert sameshape(["death_mean", "death_sd", "repr_mean",
                    "repr_sd", "fitness"], (n,pop1.maxls))
        assert sameshape(["density_surv", "density_repr"],
                    (n,2*pop1.nbase+1))
        assert sameshape(["entropy","junk_death","junk_repr",
                    "junk_fitness"], (n,))
        assert sameshape(["population_size", "resources", "surv_penf",
                    "repr_penf"], (100,))
        assert sameshape(["age_distribution"],(100,pop1.maxls))
        assert sameshape(["n1", "n1_std"], (n,pop1.chrlen))
        assert sameshape(["s1"], (n, pop1.chrlen - 99))

class TestRecordUpdate:
    """Test stage-by-stage updating of a Record object."""

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
        assert \
        np.isclose(r["death_mean"][0],
                np.tile(r["d_range"][-1],r["max_ls"])).all() and \
        np.isclose(r["death_sd"][0], np.zeros(r["max_ls"])).all() and \
        np.isclose(r["repr_mean"][0], 
                np.append(np.zeros(r["maturity"]),
                    np.tile(r["r_range"][-1],
                        r["max_ls"]-r["maturity"]))).all() and \
        np.isclose(r["repr_sd"][0], np.zeros(r["max_ls"])).all() and \
        r["density_surv"][0][-1] == 1 and \
        r["density_repr"][0][-1] == 1

    def test_update_shannon_weaver_degenerate(self,record,pop1):
        """Test if equals zero when all set members are of same type."""
        assert record.update_shannon_weaver(pop1) == -0

    def test_update_shannon_weaver(self,record,spop,conf):
        """Test that shannon weaver entropy is computed correctly for
        a newly-initialised population."""
        precision = 0.01
        b = spop.nbase
        props = np.array([spop.maxls, spop.maxls-spop.maturity, 1])\
                /float(len(spop.genmap)) # Expected proportion of genome
                # in survival loci, reproductive loci, etc.
        print props
        probs = np.array([conf.g_dist[x] for x in ["s", "r", "n"]])
                # Probability of a 1 for each locus type
        print probs
        dists = np.array(
                [[comb(2*b, x)*p**x*(1-p)**(2*b-x) for x in np.arange(2*b)+1]\
                        for p in probs])
                # Binomial distribution values for 0-20 zeros for each
        print dists
        exp = np.sum((dists.T * props).T,0) 
            # expected proportions of loci with each number of 1's over
            # entire genome
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
        assert (r["n1"][0] == np.ones(r["chr_len"])).all()
        assert (r["n1_std"][0] == np.zeros(r["chr_len"])).all()
        assert r["entropy"][0] == -0
        assert np.isclose(r["junk_death"][0], r["d_range"][-1])
        assert np.isclose(r["junk_repr"][0], r["r_range"][-1])

    def test_update(self,record,pop1):
        """Test that update properly chains quick_update,
        update_agestats and update_invstats."""
        record1 = copy.deepcopy(record)
        record.update(pop1, 100, 1, 1, 0, 0)
        record1.quick_update(0, pop1, 100, 1, 1)
        record1.update_agestats(pop1, 0)
        record1.update_invstats(pop1, 0)
        r = record.record
        r1 = record1.record
        for k in r.keys():
            assert (np.array(r[k]) == np.array(r1[k])).all()

class TestRecordFinal:
    def test_age_wise_n1(self,record):
        """Test if ten consecutive array items are correctly averaged."""
        genmap = record.record["genmap"]
        ix = np.arange(len(genmap))
        mask = np.tile(np.arange(len(genmap)).reshape((len(ix),1)),10)
        mask = mask.reshape((1,len(mask)*10))[0]
        record.record["mask"] = np.array([mask])
        assert (record.age_wise_n1("mask") == ix).all()
# not testing final_update

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

    def test_final_update(self, pop1, record):
        s = len(record.record["snapshot_stages"])
        for n in range(s):
            record.update(pop1, 100, 1, 1, 0, n)
        for m in range(100):
            record.quick_update(m, pop1, 100, 1, 1)
        record.update(pop1, 100, 1, 1, 0, 0)
        record.final_update(0, 100)
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

def test_post_cleanup():
    """Kill tempfiles made for test. Not really a test at all."""
    os.remove("sample_pop.txt")
    os.remove("sample_rec.txt")
    os.remove("log.txt")
