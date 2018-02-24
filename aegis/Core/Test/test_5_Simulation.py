from aegis.Core import Config, Population
from aegis.Core import Record, Run, Simulation
from aegis.Core import chance, init_ages, init_genomes, init_generations, deep_eq
from aegis.Core import init_gentimes, deep_key
import pytest, imp, types, random, copy, string, tempfile, os, shutil, random
import numpy as np
try:
       import cPickle as pickle
except:
       import pickle

#########################
## AUXILIARY FUNCTIONS ##
#########################

##############
## FIXTURES ##
##############

from test_1_Config import conf, conf_path, gen_trseed
from test_2a_Population_init import pop
from test_3_Record import rec, pop1, rec1
from test_4_Run import run, ran_str

@pytest.fixture(scope="module")
def sim(request, conf_path):
    """Generate an unseeded Simulation object from the default config
    test file."""
    #return Simulation(conf_path, "", 0, 100, False)
    return Simulation(conf_path, 100, False)

# TODO do also for max_fail>1
#@pytest.mark.parametrize("max_fail",[1,5])
@pytest.fixture(scope="module")
def simdf(request, conf_path):
    """Generate an unseeded Simulation object that dies off upun execution."""
    simdf = Simulation(conf_path, 100, False)
    simdf.conf["n_snapshots"] = 4
    simdf.conf["n_stages"] = 100
    simdf.conf["res_start"] = 500
    simdf.conf["startpop"] = 500
    simdf.conf["max_fail"] = 1
    simdf.conf["res_function"] = lambda n,r: r-10
    simdf.conf.generate()
    simdf.init_runs()
    return simdf

@pytest.fixture(scope="function")
def seed(request, sim):
    """Generate one Population that progressed some stages in Simulation
    and save it."""
    s = sim.copy()
    s.conf["output_mode"] = 1
    s.conf["n_runs"] = 1
    s.init_runs()
    s.execute()
    pop = [s.runs[0].record["final_pop"]]
    s.save_output()
    return pop

@pytest.fixture(scope="module")
def random_n_runs(request):
    return random.randint(2,5)

@pytest.fixture(scope="module")
def seed2(request, sim, random_n_runs):
    """Generate Populations that progressed some stages in Simulation
    and save them."""
    s = sim.copy()
    s.conf["output_mode"] = 1
    s.conf["n_runs"] = random_n_runs
    s.init_runs()
    s.execute()
    pops = [s.runs[n].record["final_pop"] for n in \
            xrange(s.conf["n_runs"])]
    s.save_output()
    return pops

###########
## TESTS ##
###########

class TestSimulationInit:
    """Test methods relating to initialising a Simulation object."""

    def test_get_conf_match(self, sim, conf):
        """Confirm that config fixture and simulation config attribute
        match, in the case that both were imported from conf_path."""
        if conf["setup"] == "import":
            sim.conf["setup"] = "import"
            conf["prng"] = sim.conf["prng"]
            conf["params"]["prng"] = sim.conf["params"]["prng"]
            # needed because sim.conf["prng"] progresses with
            # Runs init (and Pop init in every Run init) in Sim init
            # while conf["prng"] stays in state of seed
            assert sim.conf == conf

    def test_get_conf_degen(self, sim, ran_str):
        """Confirm that AEGIS raises an error if given an invalid
        config file path."""
        with pytest.raises(IOError):
            sim.get_conf(ran_str)

    def test_get_conf_good(self, sim, conf_path):
        """Test that get_conf correctly imports parameters from
        the specified path."""
        c = imp.load_source('ConfFile', conf_path)
        s = sim.copy()
        del s.conf
        s.get_conf(conf_path)
        for k in s.conf.keys():
            print k
            assert s.conf[k] == getattr(c, k)
        # TODO: Implement for multiple input files to avoid hardcoding

    def test_get_seed_degen(self, sim, ran_str):
        """Confirm that AEGIS raises an error if given an invalid
        seed file path."""
        with pytest.raises(IOError):
            sim.get_seed(ran_str)

    def test_get_seed_good(self, sim, seed):
        """Test that get_seed correctly imports a Population file."""
        opop1 = seed
        s = sim.copy()
        opop2 = s.get_seed(s.conf["output_prefix"]+\
                "_files/populations/final/run0.pop")
        shutil.rmtree(s.conf["output_prefix"]+"_files")
        assert opop1.__eq__(opop2)

    def test_get_seed_all_degen(self, sim, ran_str):
        """Confirm that AEGIS raises an error if given an invalid
        seed file path."""
        with pytest.raises(ImportError):
            sim.get_seed_all(ran_str)

    def test_get_seed_all_degen2(self, sim, random_n_runs, seed2):
        """Confirm that AEGIS raises an error if number of seed
        files and number of runs not the same."""
        with pytest.raises(ImportError):
            opops1 = seed2
            s = sim.copy()
            s.conf["n_runs"] = random_n_runs-1
            opops2 = s.get_seed_all(s.conf["output_prefix"]+\
                "_files/populations/final")

    def test_get_seed_all_good(self, sim, random_n_runs, seed2):
        opops1 = seed2
        s = sim.copy()
        s.conf["n_runs"] = random_n_runs
        opops2 = s.get_seed_all(s.conf["output_prefix"]+\
            "_files/populations/final")
        shutil.rmtree(s.conf["output_prefix"]+"_files")
        for i in xrange(len(opops1)):
            assert opops1[i].__eq__(opops2[i])

    def test_get_startpop_degen(self, sim):
        """Test that AEGIS correctly returns [""] when no seed."""
        s = sim.copy()
        s.get_startpop("")
        assert s.startpop == [""]

    def test_get_startpop_good(self, sim, seed):
        """Test that get_startpop correctly imports single file."""
        s = sim.copy()
        s.get_startpop(s.conf["output_prefix"]+\
                "_files/populations/final/run0.pop")
        shutil.rmtree(s.conf["output_prefix"]+"_files")
        opop1 = s.startpop[0]
        opop2 = seed[0]
        assert opop1.__eq__(opop2)

    def test_init_runs(self, sim):
        """Test that init_runs correctly generates new Run objects with
        the correct initial parameters."""
        s = sim.copy()
        del s.runs
        s.init_runs()
        # Test that all runs in s.runs have expected parameters
        for n in range(len(s.runs)):
            r,c = s.runs[n], s.conf.copy()
            assert isinstance(r, Run)
            assert r.log == ""
            c["genmap"] = r.conf["genmap"]
            assert c == r.conf
            assert r.surv_penf == r.repr_penf == 1.0
            assert r.n_stage == r.n_snap == 0
            assert r.n_run == n
            assert not r.dieoff
            assert not r.complete
            assert r.report_n == s.report_n
            assert r.verbose == s.verbose

class TestSimulationLogSaveAbort:
    """Test methods relating to logging, saving and aborting of a
    Simulation object."""

    def test_logprint_simulation(self, sim, ran_str):
        """Test logging (and especially newline) functionality."""
        s = sim.copy()
        s.log = ""
        s.logprint(ran_str)
        assert s.log == "{}\n".format(ran_str)

    def test_logsave(self, sim, ran_str):
        """Test saving of Simulation log based on output prefix."""
        # Setup log string
        s = sim.copy()
        s.log = ""
        s.logprint(ran_str)
        # Save log file
        outpref = os.path.join(tempfile.gettempdir(), ran_str)
        s.conf["output_prefix"] = outpref
        s.logsave()
        # Test saved file
        try:
            log_file = open(outpref + "_log.txt", "r")
            log = log_file.read()
            assert log == s.log == "{}\n".format(ran_str)
        finally:
            log_file.close()

    def test_abort(self, sim, ran_str):
        """Test Simulation aborting behaviour works correctly."""
        # Setup
        s = sim.copy()
        s.log = ""
        outpref = os.path.join(tempfile.gettempdir(), ran_str)
        s.conf["output_prefix"] = outpref
        errtype = random.choice([TypeError, ValueError, IOError])
        # Test
        with pytest.raises(errtype):
            s.abort(errtype, ran_str)
        try:
            log_file = open(outpref + "_log.txt", "r")
            log = log_file.read()
            exp = "\n{0}: {1}\n".format(errtype.__name__, ran_str)
            assert log.startswith(exp)
            assert s.log.startswith(exp)
        finally:
            log_file.close()

class TestSimulationExecution:
    """Test methods relating to execution of a Simulation object."""

    def test_execute_series(self, sim):
        """Test serial execution of simulation is equivalent to
        separately executing each run."""
        s = sim.copy()
        for r in s.runs:
            assert r.n_stage == 0
            assert not r.complete
        s.execute_series()
        for r in s.runs:
            assert r.n_stage == r.conf["n_stages"] or r.dieoff
            assert r.complete

    def test_execute_series_dieoff(self, simdf):
        """Test that simdf dies off so that first two snapshots are saved and
        the other are not."""
        s = simdf.copy()
        assert s.runs[0].n_stage == 0
        assert not s.runs[0].complete
        s.execute()
        assert s.runs[0].dieoff
        assert s.runs[0].complete
        assert s.runs[0].record["n_snapshots"] == 2

class TestSimulationFinalisation:
    """Test methods relating to finalisation and output of a Simulation
    object."""

    @pytest.mark.parametrize("output_mode", (2,1,0))
    def test_save_output_data(self, sim, ran_str, output_mode):
        """Test material (non-log) output of simulation produced by
        save_output for different output modes."""
        # Auxiliary functions
        def testfile(dirname, basename, exp_class, val=True):
            filepath = os.path.join(dirname, basename)
            # Check file exists
            ex = os.path.exists(filepath)
            assert ex if val else not ex
            # Check file contains a serialised object of expected class
            if ex:
                try:
                    f = open(filepath, "rb")
                    assert isinstance(pickle.load(f), exp_class)
                finally:
                    f.close()
        def check_output_dirs(dirpref):
            paths = ["records", "populations/final", "populations/snapshots"]
            for n in xrange(output_mode + 1):
                assert os.path.exists(os.path.join(dirpref, paths[n]))
            for m in xrange(output_mode + 1, len(paths)):
                assert not os.path.exists(os.path.join(dirpref, paths[m]))
        def confirm_output(dirpref):
            if output_mode >= 2:
                dirname = os.path.join(dirpref, "populations/snapshots")

                for n in xrange(sim.conf["n_runs"]):
                    for m in xrange(sim.conf["n_snapshots"]):
                        testfile(dirname, "run{0}_s{1}.pop".format(n,m), Population)
            if output_mode >= 1:
                dirname = os.path.join(dirpref, "populations/final")
                for n in xrange(sim.conf["n_runs"]):
                    testfile(dirname, "run{}.pop".format(n), Population)
            if output_mode >= 0:
                dirname = os.path.join(dirpref, "records")
                for n in xrange(sim.conf["n_runs"]):
                    testfile(dirname, "run{}.rec".format(n), Record)
        # Set-up
        s = sim.copy()
        outpref = os.path.join(tempfile.gettempdir(), ran_str)
        outdir = outpref + "_files"
        s.conf["output_prefix"] = outpref
        s.conf["output_mode"] = output_mode
        s.init_runs()
        s.execute()
        # Test output
        if output_mode == 2:
            for n in xrange(sim.conf["n_runs"]):
                print n, s.runs[n].record["snapshot_pops"]
        s.save_output()
        check_output_dirs(outdir)
        confirm_output(outdir)
        shutil.rmtree(outdir)
