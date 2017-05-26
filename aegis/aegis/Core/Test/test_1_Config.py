from aegis.Core import Config
import pytest,importlib,types,random,copy,string
import numpy as np

@pytest.fixture(params=["import", "random", "random"])
def conf(request):
    """Create a default configuration object."""
    c_import = importlib.import_module("config_test")
    c = Config(c_import)
    c.put("setup", request.param, "Method of fixture generation.")
    if request.param == "random": # Randomise config parameters
        # Run parameters
        c["number_of_runs"] = random.randint(1,5)
        c["number_of_stages"] = random.randint(50,400)
        c["number_of_snapshots"] = random.randint(1,20)
        c["start_pop"] = random.randint(200,1000)
        # Reproductive mode
        c["repr_mode"] = random.choice(
                ['sexual','asexual','assort_only','recombine_only'])
        # Death and reproduction parameters
        db_low, rb_low = np.random.uniform(size=2)
        db_high = db_low + random.uniform(0, 1-db_low)
        rb_high = rb_low + random.uniform(0, 1-rb_low)
        c["death_bound"] = np.array([db_low, db_high])
        c["repr_bound"] = np.array([rb_low, rb_high])
        # Mutation and recombination
        c["r_rate"], c["m_rate"], c["m_ratio"] = [random.random() for x in range(3)]
        # Genome structure
        c["g_dist_s"],c["g_dist_r"],c["g_dist_n"] = [random.random() for x in range(3)]
        c["n_neutral"] = random.randint(1, 100)
        c["n_base"] = random.randint(5, 25)
        c["repr_offset"] = random.randint(80,180)
        c["neut_offset"] = random.randint(c["repr_offset"]*3,c["repr_offset"]*5)
        # Life histories
        c["max_ls"] = random.randint(20, c["repr_offset"]-1)
        c["maturity"] = random.randint(5, c["max_ls"]-2)
        # Resources and starvation
        c["res_start"] = random.randint(500,2000)
        c["res_var"] = random.choice([True, False])
        c["res_limit"] = random.randint(3000,10000)
        c["V"] = random.randrange(10)/10.0 + random.randint(1,3)
        c["R"] = random.randint(500,2000)
        c["surv_pen"] = random.choice([True, False])
        c["repr_pen"] = random.choice([True, False])
        c["death_inc"] = random.randint(1, 10)
        c["repr_dec"] = random.randint(1, 10)
        gm_len = c["max_ls"] + (c["max_ls"] - c["maturity"]) + c["n_neutral"]
        # Sliding windows
        c["windows"] = {
                "population_size": random.randint(500,1500),
                "resources": random.randint(500,1500),
                "n1": random.randint(5,20)
                }
    # Generate derived parameters
    c.generate()
    return c

class TestConfig:

    def test_config_init(self, conf):
        """Test that Config objects are correctly initialised from an imported
        configuration file."""
        if conf["setup"] == "random": return
        c = copy.deepcopy(conf)
        # Remove stuff that gets introduced/changed during generation
        del c["g_dist"], c["genmap"], c["chr_len"], c["s_range"], c["r_range"], c["params"]
        del c["snapshot_stages"], c["surv_step"], c["repr_step"], c["genmap_argsort"]
        del c["n_states"], c["surv_bound"]
        if c["repr_mode"] in ["sexual", "assort_only"]: c["repr_bound"] /= 2
        # Compare remaining keys to directly imported config file
        c_import = importlib.import_module("config_test")
        for key in c.keys():
            if key in ["setup", "info_dict"]: continue
            assert type(c.get_info(key)) is str
            assert len(c.get_info(key)) > 0
            attr = c.get_value(key)
            impt = getattr(c_import,key)
            if type(attr) in [int, str, float, dict, bool]:
                assert attr == impt
            elif type(attr) is np.ndarray:
                assert np.array_equal(attr, impt)
            else:
                errstr = "Unexpected config element type: "
                raise TypeError(errstr + str(type(attr)))

    def test_config_check(self,conf):
        """Test that configurations with incompatible genome parameters are
        correctly rejected."""
        if conf["setup"] == "random": return
        c = copy.deepcopy(conf)
        assert c.check()
        repr_mode_old = c["repr_mode"]
        s = string.ascii_lowercase
        c["repr_mode"] = ''.join(random.choice(s) for _ in xrange(10))
        with pytest.raises(ValueError):
            c.check()
        c["repr_mode"] = repr_mode_old
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

    def test_config_getput(self,conf):
        """Test constructors and selectors for Config object."""
        if conf["setup"] == "random": return
        # Set up random inputs
        s = string.ascii_lowercase
        keystr = ''.join(random.choice(s) for _ in xrange(10))
        valstr = ''.join(random.choice(s) for _ in xrange(10))
        infostr = ''.join(random.choice(s) for _ in xrange(10))
        # Perform tests
        c = copy.deepcopy(conf)
        c.put(keystr, valstr, infostr)
        assert c.get_value(keystr) == valstr
        assert c.get_info(keystr) == infostr

    def test_config_generate(self, conf):
        """Test that gen_conf correctly generates derived simulation params."""
        c = copy.deepcopy(conf)
        np.set_printoptions(threshold=np.nan)
        # Remove stuff that gets introduced/changed during generation
        del c["g_dist"], c["genmap"], c["chr_len"], c["s_range"], c["r_range"], c["params"]
        del c["snapshot_stages"], c["surv_step"], c["repr_step"], c["genmap_argsort"]
        del c["n_states"], c["surv_bound"]
        sexvar = c["repr_mode"] in ["sexual", "assort_only"]
        if sexvar: c["repr_bound"] /= 2
        # Save info and run
        crb1 = c["repr_bound"][1]
        c.generate()
        # Test output:
        # Genome structure
        print c["max_ls"], c["maturity"], c["repr_offset"], c["neut_offset"], c["n_neutral"]
        print c["neut_offset"] - c["repr_offset"]
        print c["neut_offset"] - c["repr_offset"] < c["max_ls"]
        print c["genmap"]
        assert c["g_dist"] == {"s":c["g_dist_s"], "r":c["g_dist_r"], "n":c["g_dist_n"]}
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
        assert c["repr_bound"][1]/crb1 == 2 if sexvar else 1
        assert np.array_equal(c["surv_bound"],1-c["death_bound"][::-1])
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
        # Snapshot stages
        assert len(c["snapshot_stages"]) == c["number_of_snapshots"]
        assert np.array_equal(c["snapshot_stages"],np.around(np.linspace(
            0, c["number_of_stages"]-1, c["number_of_snapshots"]), 0))
        # Params dict
        assert type(c["params"]) is dict
        assert c["params"]["repr_mode"] == c["repr_mode"]
        assert c["params"]["chr_len"] == c["chr_len"]
        assert c["params"]["n_base"] == c["n_base"]
        assert c["params"]["maturity"] == c["maturity"]
        assert c["params"]["max_ls"] == c["max_ls"]
        assert c["params"]["start_pop"] == c["start_pop"]
        assert c["params"]["g_dist"] == c["g_dist"]
