from aegis.Core import Infodict, Config
import pytest,importlib,types,random,copy,string
import numpy as np

##############
## FIXTURES ##
##############

@pytest.fixture(scope="module")
def conf_path(request):
    return "config_test"

@pytest.fixture(params=["import", "random", "random"], scope="module")
def conf(request, conf_path):
    """Create a default configuration object."""
    c_import = importlib.import_module(conf_path)
    c = Config(c_import)
    c.put("setup", request.param, "Method of fixture generation.")
    if request.param == "random": # Randomise config parameters
        # TODO: Add seed value here?
        # Run parameters
        c["number_of_runs"] = random.randint(1,5)
        c["number_of_stages"] = random.randint(50,400)
        c["number_of_snapshots"] = random.randint(5,20)
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
        c["res_limit"] = random.randint(5000,10000)
        c["V"] = random.uniform(1,3)
        c["R"] = random.randint(500,1000)
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

####################
## INFODICT TESTS ##
####################

def ranstr(m,n=10):
    """Generate m random n-letter lowercase ASCII strings."""
    r,s = random.choice, string.ascii_lowercase
    l = [''.join(r(s) for _ in xrange(n)) for _ in xrange(m)]
    return l

class TestInfodict:

    #! TODO: Implement this
#    def test_infodict_init(self):
#        x = Infodict()
#        assert True #! ...

    def test_infodict_getput(self):
        """Test basic constructors and selectors for Infodict object."""
        # Set up random inputs
        keystr0, valstr0, infstr0 = ranstr(3)
        # Test basic put/get methods
        i = Infodict()
        i.put(keystr0, valstr0, infstr0)
        assert i.get_value(keystr0) == valstr0
        assert i.get_info(keystr0) == infstr0
        assert i[keystr0] == valstr0
        # Test single puts on existing key
        keystr1, valstr1, infstr1 = ranstr(3)
        i.put_value(keystr0,valstr1)
        i.put_info(keystr0,infstr1)
        assert i.get_value(keystr0) == valstr1
        assert i.get_info(keystr0) == infstr1
        assert i[keystr0] == valstr1
        i[keystr0] = infstr1
        assert i[keystr0] == infstr1
        # Test single puts on new key
        with pytest.raises(SyntaxError):
            i.put_value(keystr1,valstr1)
        with pytest.raises(SyntaxError):
            i.put_info(keystr1,infstr1)
        with pytest.raises(SyntaxError):
            i[keystr1] = valstr1
        # Test multiple-get methods
        i.put(keystr0, valstr0, infstr0)
        i.put(keystr1, valstr1, infstr1)
        assert i.get_values([keystr0,keystr1]) == [valstr0,valstr1]
        assert i.get_infos([keystr0,keystr1]) == [infstr0,infstr1]
        assert i[keystr0,keystr1,keystr0] == [valstr0,valstr1,valstr0]

    def test_infodict_dictmethods(self):
        """Test that dictionary methods like keys(), values() etc still work
        as expected for Infodicts."""
        keystr0, valstr0, infstr0 = ranstr(3)
        keystr1, valstr1, infstr1 = ranstr(3)
        i = Infodict()
        # keys(), values(), infos()
        i.put(keystr0, valstr0, infstr0)
        assert i.keys() == [keystr0]
        assert i.values() == [valstr0]
        assert i.infos() == [infstr0]
        assert i.has_key(keystr0)
        assert not i.has_key(keystr1)
        i.put(keystr1, valstr1, infstr1)
        assert sorted(i.keys()) == sorted([keystr0, keystr1])
        assert sorted(i.values()) == sorted([valstr0, valstr1])
        assert sorted(i.infos()) == sorted([infstr0, infstr1])
        assert i.has_key(keystr0)
        assert i.has_key(keystr1)
        # equality
        j = Infodict()
        j.put(keystr0, valstr0, infstr0)
        j.put(keystr1, valstr1, infstr1)
        assert i == i and i == j and j == j
        #! Update as more dict methods added to Infodict

    def test_infodict_delete(self):
        keystr0, valstr0, infstr0 = ranstr(3)
        keystr1, valstr1, infstr1 = ranstr(3)
        i = Infodict()
        i.put(keystr0, valstr0, infstr0)
        i.put(keystr1, valstr1, infstr1)
        i.delete_item(keystr1)
        assert i.keys() == [keystr0]
        assert i.values() == [valstr0]
        assert i.infos() == [infstr0]
        assert i.has_key(keystr0)
        assert not i.has_key(keystr1)
        i.put(keystr1, valstr1, infstr1)
        del i[keystr0]
        assert i.keys() == [keystr1]
        assert i.values() == [valstr1]
        assert i.infos() == [infstr1]
        assert i.has_key(keystr1)
        assert not i.has_key(keystr0)

    def test_infodict_subdict(self):
        # Setup
        keystr0, valstr0, infstr0 = ranstr(3)
        keystr1, valstr1, infstr1 = ranstr(3)
        keystr2, valstr2, infstr2 = ranstr(3)
        i = Infodict()
        i.put(keystr0, valstr0, infstr0)
        i.put(keystr1, valstr1, infstr1)
        i.put(keystr2, valstr2, infstr2)
        j = Infodict()
        j.put(keystr1, valstr1, infstr1)
        j.put(keystr2, valstr2, infstr2)
        # Test
        assert i.subdict([keystr0,keystr2]) == \
                {keystr0:valstr0, keystr2:valstr2}
        assert i.subdict([keystr1,keystr2],False) == j

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
