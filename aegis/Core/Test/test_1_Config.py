from aegis.Core import Config, correct_r_rate
from aegis.Core import chance, make_windows
import pytest,importlib,types,random,copy,string,os,tempfile,math,imp
import numpy as np, pickle

##############
## FIXTURES ##
##############

@pytest.fixture(scope="module")
def ran_str(request):
    """Generate a random lowercase ascii string."""
    return \
        ''.join(random.choice(string.ascii_lowercase) for _ in range(50))

@pytest.fixture(scope="module")
def gen_trseed(request):
    """Generate random seed and save it to  file trseed.
    tconfig contains a path to this file so that conf can import it."""
    f = open("./aegis/Core/Test/trseed", "w")
    rseed = np.random.RandomState().get_state()
    pickle.dump(rseed, f)
    f.close()
    return

@pytest.fixture(scope="module")
def conf_path(request, gen_trseed):
    gen_trseed
    dirpath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dirpath, "tconfig.py")
    return filepath

@pytest.fixture(params=["import", "random0", "random1", "auto0", "auto1"],\
        scope="module")
def conf_naive(request, conf_path, ran_str, gen_trseed):
    """Create a default, non-generated Config object."""
    c = Config(conf_path)
    c["setup"] = request.param
    if request.param != "import": # Randomise config parameters
        ## CORE PARAMETERS ##
        c["random_seed"] = ""
        c["output_prefix"] = os.path.join(tempfile.gettempdir(), ran_str)
        c["n_runs"] = random.randint(1,3)
        nstage = random.randint(16,80)
        c["n_stages"] = "auto" if request.param[:-1] == "auto" else nstage
        c["n_snapshots"] = random.randint(2,4)
        # c["output_mode"] = random.randrange(3) # TODO: Test with this?
        # c["max_fail"] = random.randrange(10) # TODO: Test with this?
        ## STARTING PARAMETERS ##
        c["repr_mode"] = random.choice(
                ['sexual','asexual','assort_only','recombine_only'])
        c["res_start"] = random.randint(500,1000)
        c["start_pop"] = random.randint(200,500)
        ## RESOURCE PARAMETERS ##
        a,b,d,e = np.random.uniform(size=4)
        #c["res_function"] = lambda n,r: int((r-n)*a + b) # Random affine
        c["stv_function"] = lambda n,r: e*n > d*r
        ## AUTOCOMPUTING STAGE NUMBER ##
        c["deltabar"] = random.random()*0.1
        c["scale"] = random.randint(9, 19)/10.0
        c["max_stages"] = random.randint(200,400)
        ## AGE DISTRIBUTION RECORDING ##
        if request.param[-1] == "1": age_dist_N_var = "all"
        else: age_dist_N_var = random.randint(1,3)
        c["age_dist_N"] = age_dist_N_var
        ## SIMULATION FUNDAMENTALS ##
        # Death and reproduction parameters
        sb_low, rb_low = np.random.uniform(size=2)
        sb_high = sb_low + random.uniform(0, 1-sb_low)
        rb_high = rb_low + random.uniform(0, 1-rb_low)
        c["surv_bound"] = np.array([sb_low, sb_high])
        c["repr_bound"] = np.array([rb_low, rb_high])
        # Mutation and recombination
        c["r_rate"], c["m_rate"], c["m_ratio"] = [random.random()/10 for x in range(3)]
        # Genome structure
        c["g_dist_s"],c["g_dist_r"],c["g_dist_n"] = [random.random() for x in range(3)]
        c["n_neutral"] = random.randint(1, 100)
        c["n_base"] = random.randint(5, 10)
        c["repr_offset"] = random.randint(80,180)
        c["neut_offset"] = random.randint(c["repr_offset"]*3,c["repr_offset"]*5)
        # Life histories
        c["max_ls"] = random.randint(20, c["repr_offset"]-1)
        c["maturity"] = random.randint(5, c["max_ls"]-2)
        # Resources and starvation
        gm_len = c["max_ls"] + (c["max_ls"] - c["maturity"]) + c["n_neutral"]
        # Sliding windows
        c["windows"] = {
                "population_size": random.randint(500,1500),
                "resources": random.randint(500,1500),
                "n1": random.randint(5,20)
                }
    return c

@pytest.fixture(scope="module")
def conf(request, conf_naive):
    """Create a default, generated configuration object."""
    c = copy.deepcopy(conf_naive)
    c.generate()
    return c

####################
## INFODICT TESTS ##
####################

def ranstr(m,n=10):
    """Generate m random n-letter lowercase ASCII strings."""
    r,s = random.choice, string.ascii_lowercase
    l = [''.join(r(s) for _ in xrange(n)) for _ in xrange(m)]
    return l

class TestConfig:

    def test_config_copy(self, conf, conf_naive):
        """Test that the copied config is equal for both non-generated
        and generated objects."""
        c1 = conf.copy()
        c2 = conf_naive.copy()
        assert c1 == conf
        assert c2 == conf_naive

    def test_config_auto(self, conf):
        c = conf.copy()
        assert c["auto"] == (c["n_stages"] == "auto")
        c["n_stages"] = "auto"
        c.generate()
        assert c["auto"]
        c["n_stages"] = random.randrange(100, 1000)
        c["age_dist_N"] = 40
        c["n_snapshots"] = 2
        c.generate()
        assert not c["auto"]

    def test_config_init(self, conf_naive, conf_path):
        """Test that Config objects are correctly initialised from an imported
        configuration file."""
        if conf_naive["setup"] != "import": return
        c = copy.deepcopy(conf_naive)
        # Compare remaining keys to directly imported config file
        c_import = imp.load_source('ConfFile', conf_path)
        for key in c.keys():
            if key in ["setup"]: continue
            print key
            attr = c[key]
            impt = getattr(c_import,key)
            if callable(attr):
                for n in xrange(3):
                    x = np.random.random(10)
                    n,r = np.random.uniform(size=2)
                    if attr.func_code.co_argcount == 2:
                        assert attr(n,r) == impt(n,r)
                    elif attr.func_code.co_argcount == 3:
                        assert np.array_equal(attr(x,n,r),impt(x,n,r))
            elif type(attr) in [int, str, float, dict, bool, list]:
                assert attr == impt
            elif type(attr) is np.ndarray:
                assert np.array_equal(attr, impt)
            else:
                errstr = "Unexpected config element type: "
                raise TypeError(errstr + str(type(attr)))

    def test_config_check(self,conf):
        """Test that configurations with incompatible genome parameters are
        correctly rejected."""
        if conf["setup"][:-1] == "random": return
        c = conf.copy()
        assert c.check()
        repr_mode_old = c["repr_mode"]
        s = string.ascii_lowercase
        c["repr_mode"] = ''.join(random.choice(s) for _ in xrange(10))
        with pytest.raises(ValueError):
            c.check()
        c["repr_mode"] = repr_mode_old
        res_start_old = c["res_start"]
        c["res_start"] = random.randint(-10000,-1)
        with pytest.raises(ValueError):
            c.check()
        c["res_start"] = res_start_old
        c["maturity"] = c["max_ls"] - 1
        with pytest.raises(ValueError):
            c.check()
        c["maturity"] -= 1
        c["repr_offset"] = c["max_ls"] - 1
        with pytest.raises(ValueError):
            c.check()
        c["repr_offset"] += 1
        c["neut_offset"] = c["repr_offset"] + c["max_ls"] - 1
        with pytest.raises(ValueError):
            c.check()
        c["neut_offset"] += 1
        assert c.check()
        c["age_dist_N"] = -1
        with pytest.raises(ValueError):
            c.check()
        c["age_dist_N"] = 1.1
        with pytest.raises(ValueError):
            c.check()
        c["age_dist_N"] = "notall"
        with pytest.raises(ValueError):
            c.check()

    def test_config_check2(self,conf):
        """Test that configurations with incompatible
        post-Config-generation parameters are correctly rejected."""
        if conf["setup"][:-1] == "random": return
        c = conf.copy()
        c["age_dist_N"] = 60
        c["n_stages"] = 100
        with pytest.raises(ValueError):
            c.check2()
        c["age_dist_N"] = 60
        c["n_stages"] = "auto"
        c["min_gen"] = 100
        with pytest.raises(ValueError):
            c.check2()

    def test_config_generate(self, conf_naive):
        """Test that gen_conf correctly generates derived simulation params."""
        c = copy.deepcopy(conf_naive)
        np.set_printoptions(threshold=np.nan)
        # Save info and run
        crb1 = c["repr_bound"][1]
        c.generate()
        # Prng
        assert isinstance(c["prng"],np.random.RandomState)
        # Genome structure
        print c["max_ls"], c["maturity"], c["repr_offset"], c["neut_offset"], c["n_neutral"]
        print c["neut_offset"] - c["repr_offset"]
        print c["neut_offset"] - c["repr_offset"] < c["max_ls"]
        print c["genmap"]
        assert len(c["genmap"]) == c["max_ls"] + (c["max_ls"]-c["maturity"]) +\
                c["n_neutral"]
        assert np.sum(c["genmap"] < c["repr_offset"]) == c["max_ls"]
        assert np.sum(np.logical_and(
            c["genmap"] >= c["repr_offset"],c["genmap"] < c["neut_offset"]
            )) == c["max_ls"] - c["maturity"]
        assert np.sum(c["genmap"] >= c["neut_offset"]) == c["n_neutral"]
        assert np.array_equal(np.argsort(c["genmap"]), c["genmap_argsort"])
        assert c["chr_len"] == len(c["genmap"]) * c["n_base"]
        assert c["n_states"] ==  2*c["n_base"]+1
        # Survival and reproduction
        assert np.array_equal(c["s_range"],
                np.linspace(c["surv_bound"][0],c["surv_bound"][1],
                    c["n_states"]))
        assert np.array_equal(c["r_range"],
                np.linspace(c["repr_bound"][0],c["repr_bound"][1],
                    c["n_states"]))
        assert c["surv_step"] == np.diff(c["surv_bound"])/(c["n_states"]-1)
        assert c["repr_step"] == np.diff(c["repr_bound"])/(c["n_states"]-1)
        print c["s_range"]
        print np.diff(c["s_range"])
        print c["r_range"]
        print np.diff(c["r_range"])
        assert np.allclose(np.diff(c["s_range"]), c["surv_step"])
        assert np.allclose(np.diff(c["r_range"]), c["repr_step"])
        # Params dict
        assert type(c["params"]) is dict
        assert c["params"]["repr_mode"] == c["repr_mode"]
        assert c["params"]["chr_len"] == c["chr_len"]
        assert c["params"]["n_base"] == c["n_base"]
        assert c["params"]["maturity"] == c["maturity"]
        assert c["params"]["max_ls"] == c["max_ls"]
        assert c["params"]["start_pop"] == c["start_pop"]
        assert c["params"]["g_dist"] == c["g_dist"]
        # Snapshot stages
        if c["auto"]:
            alpha, beta = c["m_rate"], c["m_rate"]*c["m_ratio"]
            delta = c["deltabar"]*beta/(alpha+beta)
            y = c["g_dist"]["n"]
            x = 1-y
            k = np.log(delta*(alpha+beta)/abs(alpha*y-beta*x)) / \
                    np.log(abs(1-alpha-beta))
            assert c["min_gen"] == int(k * c["scale"])
            assert len(c["snapshot_generations"]) == c["n_snapshots"]
            assert len(c["snapshot_generations"]) == c["n_snapshots"]
            assert c["snapshot_generations"].dtype is np.dtype(int)
            assert np.array_equal(c["snapshot_generations"],np.around(
                np.linspace(0, c["min_gen"], c["n_snapshots"])))
            assert np.array_equal(c["snapshot_generations"],
                    c["snapshot_generations_remaining"])
        else:
            assert len(c["snapshot_stages"]) == c["n_snapshots"]
            assert c["snapshot_stages"].dtype is np.dtype(int)
            assert np.array_equal(c["snapshot_stages"],np.around(np.linspace(
                0, c["n_stages"]-1, c["n_snapshots"])))
        # Crossover rates
        assert c["r_rate_input"] == conf_naive["r_rate"]
        assert c["r_rate"] == correct_r_rate(conf_naive["r_rate"])
        # Age distribution windows
        if c["age_dist_N"] != "all":
            if c["auto"]:
                assert np.array_equal(c["age_dist_generations"],\
                        make_windows(c["snapshot_generations"], c["age_dist_N"]))
            else:
                assert np.array_equal(c["age_dist_stages"],\
                        make_windows(c["snapshot_stages"], c["age_dist_N"]))
