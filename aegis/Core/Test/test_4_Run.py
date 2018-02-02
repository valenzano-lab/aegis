from aegis.Core import Config, Population, Record, Run
from aegis.Core import chance, init_ages, init_genomes, init_generations, deep_eq
from aegis.Core import init_gentimes
import pytest,importlib,types,random,copy,string
import numpy as np

#########################
## AUXILIARY FUNCTIONS ##
#########################

##############
## FIXTURES ##
##############

from test_1_Config import conf, conf_path, ran_str
from test_2a_Population_init import pop
from test_3_Record import rec, pop1, rec1

@pytest.fixture(scope="module")
def run(request, conf):
    """Unseeded run object inheriting from conf fixture."""
    return Run(conf, "", 0, 100, False)

###########
## TESTS ##
###########

class TestRun:

    @pytest.mark.parametrize("report_n, verbose",
            [(random.randint(1, 100), True), (random.randint(1, 100), False)])
    def test_init_run(self, conf, report_n, verbose):
        run1 = Run(conf, "", conf["n_runs"]-1, report_n, verbose)
        assert run1.log == ""
        assert run1.surv_penf == run1.repr_penf == 1.0
        assert run1.resources == conf["res_start"]
        assert sorted(run1.genmap) == sorted(conf["genmap"])
        assert not np.array_equal(run1.genmap, conf["genmap"])
        #! TODO: Test correct inheritance of conf vs population parameters
        assert run1.n_snap == run1.n_stage == 0
        assert run1.n_run == conf["n_runs"]-1
        #! TODO: Test initial state of record vs conf?
        assert np.array_equal(run1.record["genmap"], run1.genmap)
        assert run1.dieoff == run1.complete == False
        assert run1.report_n == report_n
        assert run1.verbose == verbose
        # Quick test of correct genmap transition from run -> pop -> record;
        # Population and Record initiation tested more thoroughly elsewhere

    def test_update_resources(self, run):
        """Test resource updating between bounds and confirm resources
        cannot go outside them."""
        run1 = run.copy()
        rl,rf = run1.conf["res_limit"], run1.conf["res_function"]
        def rtest():
            old_r = run1.resources
            run1.update_resources()
            assert run1.resources == min(rl,max(0,rf(run1.population.N,old_r)))
        ## Test conditions (initial, overgrowth, undergrowth, midpoint)
        cond = [run1.resources, rl * 2, -rl, rl / 2]
        for c in cond:
            run1.resources = c
            rtest()

    def test_starving(self, run):
        """Test run enters starvation state under correct conditions."""
        run1 = run.copy()
        # 1: Deterministic test
        if run1.conf["setup"] == "import":
            test = {-1:True, 0:False, 1:False}
            for c in test.keys():
                run1.resources = run1.population.N + c
                assert run1.starving() == test[c]
        # 2: Random test
        for n in xrange(3):
            r,n = np.random.uniform(0, run1.conf["res_limit"], size=2)
            run1.population.N, run1.resources = n,r
            assert run1.starving() == run1.conf["stv_function"](n,r)

    @pytest.mark.parametrize("spen", [True, False])
    @pytest.mark.parametrize("rpen", [True, False])
    def test_update_starvation_factors(self, run, spen, rpen):
        """Test that starvation factors update correctly under various
        conditions for standard starvation function."""
        run1 = run.copy()
        run1.conf["surv_pen"], run1.conf["repr_pen"] = spen, rpen
        run1.conf["stv_function"] = lambda n,r: n > r
        # Expected changes
        ec_s = run1.conf["death_inc"] if spen else 1.0
        ec_r = run1.conf["repr_dec"]  if rpen else 1.0
        # 1: Under non-starvation, factors stay at 1.0
        run1.resources = run1.population.N + 1
        run1.update_starvation_factors()
        assert run1.surv_penf == run1.repr_penf == 1.0
        # 2: Under starvation, factors increase
        run1.resources = run1.population.N - 1
        run1.update_starvation_factors()
        assert run1.surv_penf == ec_s
        assert run1.repr_penf == ec_r
        # 3: Successive starvation compounds factors exponentially
        run1.update_starvation_factors()
        assert run1.surv_penf == ec_s**2
        assert run1.repr_penf == ec_r**2
        # 4: After starvation ends factors reset to 1.0
        run1.resources = run1.population.N + 1
        run1.update_starvation_factors()
        assert run1.surv_penf == 1.0
        assert run1.repr_penf == 1.0

    def test_execute_stage_functionality(self, run):
        """Test basic functional operation of test_execute_stage, ignoring
        status reporting."""
        run1 = run.copy()
        # Normal
        run1.execute_stage()
        assert run1.n_stage == run.n_stage + 1
        assert run1.n_snap == run.n_snap + 1
        assert (run1.dieoff == (run1.population.N == 0))
        assert run1.complete == run1.dieoff
        run1.execute_stage()
        assert run1.n_stage == run.n_stage + 2
        assert run1.n_snap == run.n_snap + 1
        assert (run1.dieoff == (run1.population.N == 0))
        assert run1.complete == run1.dieoff
        #! TODO: Add tests of age incrementation, record updating, growth, death
        # Last stage
        run2 = run.copy()
        n = 0
        while not run2.complete:
            print run2.n_stage, run2.population.N, len(run2.population.generations),
            print np.min(run2.population.generations),
            print run2.conf["min_gen"] if run2.conf["auto"] else ""
            run2.execute_stage()
        assert run2.complete
        assert (run2.dieoff == (run2.population.N == 0))
        print run2.n_stage, run2.n_snap
        if not run2.dieoff: # Run completion
            assert run2.n_snap == run.conf["n_snapshots"]
            assert run2.n_stage == run.conf["n_stages"]
        elif not run2.conf["auto"]: # Set stage count + dieoff
            print run2.conf["snapshot_stages"]
            assert run2.n_snap == 1+np.max(
                    np.nonzero(run2.conf["snapshot_stages"]<run2.n_stage)[0])
        else: # Auto stage count + dieoff
            print run2.conf["snapshot_generations"]
            print run2.conf["snapshot_generations_remaining"]
            assert run2.n_snap == run.conf["n_snapshots"] - \
                    len(run2.conf["snapshot_generations_remaining"])
        # Dead
        run3 = run.copy()
        run3.population = run3.population
        run3.population.N = 0
        run3.population.ages = np.array([])
        run3.population.genomes = np.array([[],[]])
        run3.execute_stage()
        assert run3.n_stage == run.n_stage + 1
        assert run3.n_snap == run.n_snap
        assert run3.dieoff and run3.complete
        #! TODO: Add test for status reporting?

    #! TODO: Add test for Run.execute (that will actually run...)

    def test_logprint_run(self, run, ran_str):
        """Test logging (and especially newline) functionality."""
        R2 = run.copy()
        R2.log = ""
        R2.conf["n_runs"] = 1
        R2.conf["n_stages"] = 1
        R2.n_run = 0
        R2.n_stage = 0
        R2.logprint(ran_str)
        assert R2.log == "RUN 0 | STAGE 0 | {0}\n".format(ran_str)
        R2.log = ""
        R2.conf["n_runs"] = 101
        R2.conf["n_stages"] = 101
        R2.logprint(ran_str)
        assert R2.log == "RUN   0 | STAGE   0 | {0}\n".format(ran_str)
