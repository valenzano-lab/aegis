from aegis.Core import Infodict, Config, Population, Outpop, Record, Run
from aegis.Core import chance, init_ages, init_genomes, init_generations, deepeq
import pytest,importlib,types,random,copy,string
import numpy as np

#########################
## AUXILIARY FUNCTIONS ##
#########################

##############
## FIXTURES ##
##############

from test_1_Config import conf, conf_path
from test_2a_Population_init import pop
from test_3_Record import rec, pop1, rec1

@pytest.fixture(scope="module")
def run(request, conf):
    """Unseeded run object inheriting from conf fixture."""
    return Run(conf, "", 0, 100, False)

@pytest.fixture()
def ran_str(request, scope="module"):
    """Generate a random lowercase ascii string."""
    return \
        ''.join(random.choice(string.ascii_lowercase) for _ in range(50))

###########
## TESTS ##
###########

# TODO: Test seeding

class TestRun:

    @pytest.mark.parametrize("report_n, verbose",
            [(random.randint(1, 100), True), (random.randint(1, 100), False)])
    def test_init_run(self, conf, report_n, verbose):
        run1 = Run(conf, "", conf["number_of_runs"]-1, report_n, verbose)
        assert run1.log == ""
        assert run1.surv_penf == run1.repr_penf == 1.0
        assert run1.resources == conf["res_start"]
        assert sorted(run1.genmap) == sorted(conf["genmap"])
        assert not np.array_equal(run1.genmap, conf["genmap"])
        #! TODO: Test correct inheritance of conf vs population parameters
        assert run1.n_snap == run1.n_stage == 0
        assert run1.n_run == conf["number_of_runs"]-1
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
        r,v,rl = run1.conf["R"], run1.conf["V"], run1.conf["res_limit"]
        print r,v,rl
        def R(val=-1): 
            if val > 0: run1.resources = val
            else: return run1.resources
        def N(val=-1):
            if val > 0: run1.population.N = val
            else: return run1.population.N
        ## Constant resources
        run1.conf["res_var"] = False
        old_res = run1.resources
        run1.update_resources()
        assert R() == old_res
        ## Variable resources
        run1.conf["res_var"] = True
        # State 1: R > N, regrowth hits limit
        R(random.randint(rl-r/2,rl)); N(random.randint(0, r/2))
        run1.update_resources()
        assert R() == rl
        # State 2: R > N, regrowth stays  within limits
        print r+1, int((rl-r)/v)
        R(random.randint(r+1, int((rl-r)/v)))
        N(random.randint(int((R()-r)/v), int(R()-1)))
        r_old = R()
        run1.update_resources()
        assert R() == int((r_old - N())*v + r)
        # State 3: R < N, difference > R
        R(random.randint(0, rl-r)); N(random.randint(int(R()+r), rl))
        run1.update_resources()
        assert R() == 0
        # State 4: R < N, difference < R
        R(random.randint(0, rl-r)); N(random.randint(R(), int(R()+r)))
        r_old = R()
        run1.update_resources()
        assert R() == int((r_old - N()) + r)

    def test_starving(self, run):
        """Test run enters starvation state under correct conditions for
        constant and variable resources."""
        run1 = run.copy()
        # Constant resources
        run1.conf["res_var"] = False
        run1.resources, run1.population.N = 5000, 4999
        assert not run1.starving()
        run1.resources, run1.population.N = 4999, 5000
        assert run1.starving()
        # Variable resources
        run1.conf["res_var"] = True
        run1.resources = 1
        assert not run1.starving()
        run1.resources = 0
        assert run1.starving()

    @pytest.mark.parametrize("spen", [True, False])
    @pytest.mark.parametrize("rpen", [True, False])
    def test_update_starvation_factors(self, run, spen, rpen):
        """Test that starvation factors update correctly under various
        conditions."""
        run1 = run.copy()
        run1.conf["surv_pen"], run1.conf["repr_pen"] = spen, rpen
        # Expected changes
        ec_s = run1.conf["death_inc"] if spen else 1.0
        ec_r = run1.conf["repr_dec"]  if rpen else 1.0
        # 1: Under non-starvation, factors stay at 1.0
        run1.conf["res_var"], run1.resources = True, 1
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
        assert run1.surv_penf == 1.0
        assert run1.repr_penf == 1.0

    def test_execute_stage_functionality(self, run):
        """Test basic functional operation of test_execute_stage, ignoring
        status reporting."""
        # Normal
        run1 = run.copy()
        run1.population = run1.population.toPop()
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
        run2.population = run2.population.toPop()
        print run2.conf["snapshot_stages"]
        print run2.n_stage, run2.n_snap
        n = 0
        while not run2.complete:
            run2.execute_stage()
        assert (run2.dieoff == (run2.population.N == 0))
        if not run2.dieoff:
            assert run2.n_stage == run.conf["number_of_stages"]
            assert run2.n_snap == run.conf["number_of_snapshots"]
        else:
            print run2.n_stage, run2.conf["snapshot_stages"]
            assert run2.n_snap == 1+np.max(
                    np.nonzero(run2.conf["snapshot_stages"]<=run2.n_stage)[0])
        assert run2.complete
        # Dead
        run3 = run.copy()
        run3.population = run3.population.toPop()
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
        R2.conf["number_of_runs"] = 1
        R2.conf["number_of_stages"] = 1
        R2.n_run = 0
        R2.n_stage = 0
        R2.logprint(ran_str)
        assert R2.log == "RUN 0 | STAGE 0 | {0}\n".format(ran_str)
        R2.log = ""
        R2.conf["number_of_runs"] = 101
        R2.conf["number_of_stages"] = 101
        R2.logprint(ran_str)
        assert R2.log == "RUN   0 | STAGE   0 | {0}\n".format(ran_str)
