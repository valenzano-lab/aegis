from aegis.Core import Infodict, Config, Population, Outpop
from aegis.Core import Record, Run, Simulation
from aegis.Core import chance, init_ages, init_genomes, init_generations, deepeq
from aegis.Core import init_gentimes
import pytest, imp, types, random, copy, string, tempfile, os, shutil
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

from test_1_Config import conf, conf_path
from test_2a_Population_init import pop
from test_2d_Outpop import opop
from test_3_Record import rec, pop1, rec1
from test_4_Run import run, ran_str


@pytest.fixture(scope="module")
def sim(request, conf_path):
    """Generate an unseeded Simulation object from the default config
    test file."""
    #return Simulation(conf_path, "", 0, 100, False)
    return Simulation(conf_path, 100, False)

###########
## TESTS ##
###########

class TestSimulationInit:
    """Test methods relating to initialising a Simulation object."""

    def test_get_conf_match(self, sim, conf):
        """Confirm that config fixture and simulation config attribute
        match, in the case that both were imported from conf_path."""
        if conf["setup"] == "import":
            sim.conf.put("setup", "import", "Method of fixture generation.")
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

    # TODO: Expert sample objects for testing seed setting and get_startpop
    # def test_get_seed(self, ...)
    #def test_get_startpop(self, ...)

    def test_get_startpop_degen(self, sim):
        s = sim.copy()
        for n in xrange(3):
            s.get_startpop("", random.randint(-1e6,1e6))
            assert s.startpop == [""]

    def test_set_startpop_opop_pop(self, sim, opop, pop):
        """Test set_startpop functionality for Population and Outpop
        seed objects."""
        s1 = sim.copy()
        del s1.startpop
        s2 = s1.copy()
        s3 = s1.copy()
        # 1: Outpop
        s1.set_startpop(opop, -1)
        s2.set_startpop(opop, random.randint(-1e6, 1e6))
        s3.set_startpop(opop, random.random())
        assert s1.startpop == opop
        assert s2.startpop == opop
        assert s3.startpop == opop
        del s1.startpop, s2.startpop, s3.startpop
        # 2: Population
        s1.set_startpop(pop, -1)
        s2.set_startpop(pop, random.randint(-1e6, 1e6))
        s3.set_startpop(pop, random.random())
        assert s1.startpop == opop
        assert s2.startpop == opop
        assert s3.startpop == opop

    # TODO: Add tests for setting startpops from Record, Run and
    #       Simulation objects
#    def test_set_startpop(self, sim, opop, pop, rec, rec1, run):
#        """Test set_startpop functionality for different classes and
#        parameter values."""
#        print vars(sim)
#        print sim.startpop
#
#        # 3: Record (failed)
#        with pytest.raises(ValueError):
#            sim.set_startpop(rec, -1)
#        with pytest.raises(ValueError):
#            sim.set_startpop(rec, rec["number_of_snapshots"]+1)
#        ao.uao.eoe
#        # 4: Run
#        sim.set_startpop(run, -1)
#        sim2.set_startpop(run, random.randint(-1e6, -1))
#        assert sim.startpop == run.pop
#        assert sim2.startpop == run.pop
#        assert 1 == 2

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

    def test_startpop_simulation(self, sim):
        """Test that the __startpop__ method for the Simulation class
        behaves appropriately."""
        for n in xrange(2): # Check that process fails if input is -ve
            x = sim.__startpop__(random.randint(-1e6,-1))
            assert x[0] == ValueError
            m = random.randrange(sim.conf["number_of_runs"])
            y = sim.__startpop__(m)
            assert y[0] == sim.runs[m].__startpop__(-1)[0]

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
            assert r.n_stage == r.conf["number_of_stages"] or r.dieoff
            assert r.complete

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
                for n in xrange(sim.conf["number_of_runs"]):
                    for m in xrange(sim.conf["number_of_snapshots"]):
                        testfile(dirname, "run{0}_s{1}.pop".format(n,m), Outpop)
            if output_mode >= 1:
                dirname = os.path.join(dirpref, "populations/final")
                for n in xrange(sim.conf["number_of_runs"]):
                    testfile(dirname, "run{}.pop".format(n), Outpop)
            if output_mode >= 0:
                dirname = os.path.join(dirpref, "records")
                for n in xrange(sim.conf["number_of_runs"]):
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
            for n in xrange(sim.conf["number_of_runs"]):
                print n, s.runs[n].record["snapshot_pops"]
        s.save_output()
        check_output_dirs(outdir)
        confirm_output(outdir)
        shutil.rmtree(outdir)
