from aegis.Core import Config, Population, Record, Run
from aegis.Core import chance, init_ages, init_genomes, init_generations, deep_eq
from aegis.Core import init_gentimes
import pytest,importlib,types,random,copy,string,os
import numpy as np, cPickle as pickle

#########################
## AUXILIARY FUNCTIONS ##
#########################

##############
## FIXTURES ##
##############

from test_1_Config import conf, conf_naive, conf_path, ran_str, gen_trseed
from test_2a_Population_init import pop
from test_3_Record import rec, pop1, rec1

@pytest.fixture(scope="module")
def run(request, conf):
    """Unseeded run object inheriting from conf fixture."""
    return Run(conf, "", 0, 100, False)

# pop is initialized with conf
@pytest.fixture(scope="module")
def conf_seed(request, conf):
    c = copy.copy(conf)
    for key in ["n_neutral", "n_base", "repr_offset", "neut_offset", "max_ls", "maturity", "chr_len"]:
        c[key] += 1
    return c

@pytest.fixture(params=["noauto_all","noauto-nodieoff","noauto-dieoff", \
        "auto-nodieoff","auto-dieoff", "auto-max_stages"], scope="module")
def confx(request, conf_path):
    c = Config(conf_path)
    c["setup"] = request.param
    c["res_start"] = c["start_pop"] = 300
    c["n_snapshots"] = 5
    c["n_stages"] = 1000
    c["output_mode"] = 0
    c["age_dist_N"] = 10
    c["max_fail"] = 1
    c["kill_at"] = 0
    c["max_stages"] = 50000
    if request.param == "noauto_all":
        c["age_dist_N"] = "all"
        c["n_snapshot"] = 2
        c.generate()
    elif request.param == "auto-nodieoff":
        c["m_rate"] = 0.02
        c["m_ratio"] = 0.80
        c["n_stages"] = "auto"
        c.generate()
    elif request.param == "auto-max_stages":
        c["m_rate"] = 0.02
        c["m_ratio"] = 0.80
        c["n_stages"] = "auto"
        c["max_stages"] = 200
        print "age_dist_N: ", c["age_dist_N"]
        print "n snaps: ", c["n_snapshots"]
        c.generate()
    elif request.param == "auto-dieoff":
        c["m_rate"] = 0.02
        c["m_ratio"] = 0.80
        c["n_stages"] = "auto"
        c["max_fail"] = 2
        c.generate()
        c["kill_at"] = c["snapshot_generations"][1]+1
    elif request.param == "noauto-dieoff":
        c["max_fail"] = 3
        c.generate()
        c["kill_at"] = np.mean((c["snapshot_stages"][1],\
                c["snapshot_stages"][2])).astype(int)
    else:
        c.generate()
    return c

@pytest.fixture(scope="module")
def runx(request, confx):
    return Run(confx, "", 0, 100, False)

###########
## TESTS ##
###########

class TestRun:

    @pytest.mark.parametrize("report_n, verbose",
            [(random.randint(1, 100), True), (random.randint(1, 100), False)])
    def test_init_run(self, conf, report_n, verbose):
        run1 = Run(conf, "", conf["n_runs"]-1, report_n, verbose)
        assert run1.log == ""
        assert np.array_equal(run1.s_range,conf["s_range"])
        assert np.array_equal(run1.r_range,conf["r_range"])
        assert run1.resources == conf["res_start"]
        assert sorted(run1.genmap) == sorted(conf["genmap"])
        assert not np.array_equal(run1.genmap, conf["genmap"])
        assert run1.n_snap == run1.n_stage == 0
        assert run1.n_run == conf["n_runs"]-1
        # TODO: Test initial state of record vs conf?
        assert np.array_equal(run1.record["genmap"], run1.genmap)
        assert run1.dieoff == run1.complete == False
        assert run1.report_n == report_n
        assert run1.verbose == verbose
        # Quick test of correct genmap transition from run -> pop -> record;
        # Population and Record initiation tested more thoroughly elsewhere

    def test_init_run_seed_pop(self, conf_seed, pop):
        """Test that run initialization with a seed population, defualts to parameter values from the seed population when
        there is a conflict (for gene structure specific parameters)."""
        pop_params = pop.params()
        with pytest.warns(UserWarning):
            run1 = Run(conf_seed, pop, conf_seed["n_runs"]-1, report_n=10, verbose=False)
        for key in ["n_neutral", "n_base", "repr_offset", "neut_offset", "max_ls", "maturity", "chr_len"]:
            assert run1.conf[key] == pop_params[key]
            assert run1.conf[key] != conf_seed[key]

    def test_copy(self, run):
        """Test that copy of object is equal to the object."""
        run1 = run.copy()
        assert run == run1

    def test_update_resources(self, run):
        """Test resource updating between bounds and confirm resources
        cannot go outside them."""
        run1 = run.copy()
        rfs = [lambda n,r: r, lambda n,r: 2*r, lambda n,r: 0.9*r, lambda n,r:r+1]
        oldr = run1.resources
        exps = [oldr, 2*oldr, 0.9*2*oldr, 0.9*2*oldr+1]
        for rf,exp in zip(rfs,exps):
            run1.conf["res_function"] = rf
            run1.update_resources()
            assert run1.resources == exp

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
            r,n = np.random.uniform(0, run1.resources, size=2)
            run1.population.N, run1.resources = n,r
            assert run1.starving() == run1.conf["stv_function"](n,r)

    @pytest.mark.parametrize("pen_cuml", [True, False])
    @pytest.mark.parametrize("surv_pen", [lambda sr,n,r:sr, lambda sr,n,r:sr*0.9])
    @pytest.mark.parametrize("repr_pen", [lambda rr,n,r:rr, lambda rr,n,r:rr*0.9])
    def test_update_starvation_factors(self, run, pen_cuml, surv_pen, repr_pen):
        """Test that starvation factors update correctly under various
        conditions for standard starvation function."""
        run1 = run.copy()
        run1.conf["pen_cuml"] = pen_cuml
        run1.conf["stv_function"] = lambda n,r: n > r
        run1.conf["surv_pen_func"] = surv_pen
        run1.conf["repr_pen_func"] = repr_pen
        # Expected changes
        ec_s = run1.conf["surv_pen_func"](run1.conf["s_range"], run1.population.N,\
                run1.resources)
        ec_s2 = run1.conf["surv_pen_func"](ec_s, run1.population.N,\
                run1.resources)
        ec_r = run1.conf["repr_pen_func"](run1.conf["r_range"], run1.population.N,\
                run1.resources)
        ec_r2 = run1.conf["repr_pen_func"](ec_r, run1.population.N,\
                run1.resources)
        # 1: Under non-starvation, factors stay default
        run1.resources = run1.population.N + 1
        run1.update_starvation_factors()
        assert np.array_equal(run1.s_range,run1.conf["s_range"])
        assert np.array_equal(run1.r_range,run1.conf["r_range"])
        # 2: Under starvation, factors increase
        run1.resources = run1.population.N - 1
        run1.update_starvation_factors()
        assert np.array_equal(run1.s_range,ec_s)
        assert np.array_equal(run1.r_range,ec_r)
        # 3: Successive starvation compounds function if pen_cuml
        run1.update_starvation_factors()
        if pen_cuml:
            assert np.array_equal(run1.s_range,ec_s2)
            assert np.array_equal(run1.r_range,ec_r2)
        else:
            assert np.array_equal(run1.s_range,ec_s)
            assert np.array_equal(run1.r_range,ec_r)
        # 4: After starvation ends factors reset to default
        run1.resources = run1.population.N + 1
        run1.update_starvation_factors()
        print "starving ",run1.starving()
        assert np.array_equal(run1.s_range,run1.conf["s_range"])
        assert np.array_equal(run1.r_range,run1.conf["r_range"])

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
        # TODO: Add tests of age incrementation, record updating, growth, death
        # Last stage
        run2 = run.copy()
        if run2.conf["auto"]:
            run2.conf["min_gen"] = 4
            run2.conf["snapshot_generations_remaining"] = np.around(np.linspace(0,4,\
                    run2.conf["n_snapshots"])).astype(int)
        while not run2.complete:
            print run2.n_stage, run2.population.N, len(run2.population.generations),
            print np.min(run2.population.generations),
            print run2.conf["min_gen"] if run2.conf["auto"] else ""
            run2.execute_stage()
        assert run2.complete
        assert (run2.dieoff == (run2.population.N == 0))
        print run2.n_stage, run2.n_snap
        if not run2.dieoff and not run2.conf["auto"]: # Run completion
            assert run2.n_snap == run.conf["n_snapshots"]
            assert run2.n_stage == run.conf["n_stages"]
        elif not run2.conf["auto"]: # Set stage count + dieoff
            print run2.conf["snapshot_stages"]
            assert run2.n_snap == 1+np.max(
                    np.nonzero(run2.conf["snapshot_stages"]<run2.n_stage-1)[0])
        else: # Auto stage count + dieoff
            print run2.conf["snapshot_generations"]
            print run2.conf["snapshot_generations_remaining"]
            assert run2.n_snap == run.conf["n_snapshots"] - \
                    len(run2.conf["snapshot_generations_remaining"])
        # Dead
        run3 = run.copy()
        mask = np.zeros(run3.population.N, dtype=bool)
        run3.population = run3.population.subset_clone(mask)
        run3.execute_stage()
        assert run3.n_stage == run.n_stage + 1
        assert run3.population.N == 0
        assert run3.n_snap == run.n_snap
        assert run3.dieoff and run3.complete
        # TODO: Add test for status reporting?

    def test_test_complete(self, run):
        """Test that completion is correctly identified in auto,
        non-auto, and dieoff cases."""
        # Setup test run
        run1 = run.copy()
        assert not run1.dieoff
        assert not run1.complete
        # Auxiliary function
        def check_complete(s=True):
            run1.test_complete()
            assert run1.complete == s
        # Tests
        # 1: Dieoff case
        for x in [True, False]:
            run1.dieoff = x
            check_complete(x)
        # 2: No dieoff, non-auto
        if not run1.conf["setup"][:-1] == "auto":
            assert not hasattr(run1, "min_gen")
            assert run1.n_stage < run1.conf["n_stages"]
            for n in [0, 1, 0]:
                run1.n_stage = run1.conf["n_stages"] - 1 + n
                check_complete(n)
        # 3: No dieoff, auto
        else:
            assert run1.n_stage < run1.conf["max_stages"]
            assert np.min(run1.population.generations) < run1.conf["min_gen"]
            for n in [0, 1, 0]: # Stage requirement only
                run1.n_stage = run1.conf["max_stages"] - 1 + n
                check_complete(n)
            for m in [0, 1, 0]: # Gen requirement only
                run1.population.generations[:] = run1.conf["min_gen"] - 1 + n
                check_complete(n)
            # Both requirements together
            run1.n_stage = run1.conf["max_stages"] + 1
            run1.population.generations[:] = run1.conf["min_gen"] + 1
            check_complete(True)

    def test_execute_attempt_completion(self, run):
        """Test that a given run attempt is correctly executed to
        completion under execute_attempt."""
        # Test normal execution
        run1 = run.copy()
        assert not run1.complete
        run1.execute_attempt()
        assert run1.complete # Check automatic completion
        run1.complete = False
        run1.test_complete() # Check completion conditions
        assert run1.complete
        # Force dieoff
        run2 = run.copy()
        run2.population.subtract_members(np.arange(run2.population.N))
        assert run2.population.N == 0 # Check pop correctly emptied
        assert not run2.complete
        run2.execute_attempt()
        assert run2.dieoff
        assert run2.complete

    def test_execute_attempt_starttime(self, run):
        """Test that execute_attempt generates a new start time for a
        run if one is absent, but keeps the old one otherwise."""
        run1 = run.copy()
        # Generate initial start time
        assert not hasattr(run1, "starttime")
        run1.execute_attempt()
        assert hasattr(run1, "starttime")
        run2 = run.copy()
        start = run2.starttime = run1.starttime
        run2.execute_attempt()
        assert run1.starttime == start

    def test_execute_baseline(self, run):
        """Test, as a basic requirement, that after execution a run is
        complete and finalised correctly."""
        run1 = run.copy()
        assert not run1.complete
        assert not run1.record["finalised"]
        assert run1.record["prev_failed"] == 0
        run1.execute()
        assert run1.complete
        assert run1.record["finalised"]
        print run1.record["prev_failed"], "/", run1.record["max_fail"]
        if not run1.dieoff:
            assert run1.record["prev_failed"] < run1.record["max_fail"]
        else:
            assert run1.record["prev_failed"] == run1.record["max_fail"] - 1

    def test_execute_repeats(self, run):
        """Test repeat functionality for execute under controlled
        dieoff conditions."""
        if run.conf["setup"] != "import": return
        maxfail = run.record["max_fail"]
        for M in np.arange(maxfail):
            # Initialise test run
            run1 = run.copy()
            #run1.verbose=True
            #run1.report_n = 1
            assert not run1.complete
            assert not run1.record["finalised"]
            assert run1.record["prev_failed"] == 0
            # Define resource function
            res_function = run1.conf["res_function"]
            def induced_death_resources(n,r):
                """Force dieoff through starvation in first n
                attempts."""
                if run1.record["prev_failed"] < M:
                    return 0
                else:
                    return res_function(n,r)
            run1.conf["res_function"] = induced_death_resources
            # Run and test
            run1.execute()
            assert run1.complete
            assert run1.record["finalised"]
            print M, ":", run1.record["prev_failed"], "/", maxfail
            assert run1.record["prev_failed"] == min(M,maxfail)

    def test_execute(self, runx):
        """Test that age distribution is correctly recorded and data truncated
        in different scenarios."""
        R = runx.copy()
        nsnap = copy.copy(R.record["n_snapshots"])
        nstage = R.n_stage
        maxls = R.record["max_ls"]
        adN = R.record["age_dist_N"]
        # setup
        R.record["population_size"][:] = R.population.N
        n = R.conf["n_stages"] if not R.conf["auto"] else R.conf["max_stages"]
        R.record["age_distribution"] = np.random.random((n,maxls))
        # penalties, generation_dist, gentime_dist unchanged
        if R.conf["setup"] == "noauto_all":
            for i in xrange(nsnap):
                R.record["snapshot_pops"][i] = R.population.clone()
            R.n_stage = R.conf["n_stages"]-1
            R.execute()
            assert not R.dieoff
            assert R.record["age_distribution"].shape == (R.conf["n_stages"],maxls)
        elif R.conf["setup"] == "noauto-nodieoff":
            for i in xrange(nsnap):
                R.record["snapshot_pops"][i] = R.population.clone()
            R.n_stage = R.conf["n_stages"]-1
            R.execute()
            assert not R.dieoff
            assert R.record["age_distribution"].shape == (nsnap,
                    R.conf["age_dist_N"],maxls)
        elif R.conf["setup"] == "noauto-dieoff":
            # dieoff between second and third snapshot
            for i in xrange(2):
                R.record["snapshot_pops"][i] = R.population.clone()
            R.n_stage = R.conf["kill_at"]-1
            R.execute()
            assert R.dieoff
            assert R.record["n_snapshots"] == 2
            assert R.record["age_distribution"].shape == (2,
                    R.conf["age_dist_N"],maxls)
            # check dieoff info recording
            assert R.record["dieoff_at"].shape == (R.conf["max_fail"],)
            assert np.all(R.record["dieoff_at"] != 0)
            assert np.allclose(R.record["dieoff_at"], R.conf["kill_at"],\
                    atol=10)
        elif R.conf["setup"] == "auto-nodieoff":
            # for last snapshot check age_dist_stages updated
            for i in xrange(nsnap):
                R.record["snapshot_pops"][i] = R.population.clone()
            a = np.linspace(0,R.conf["max_stages"]/2,nsnap).astype(int)
            minl = R.conf["max_stages"]/2
            for i in xrange(nsnap-1):
                x1 = x2 = 0
                while x1==x2:
                    x1 = np.random.randint(a[i],a[i+1]-1)
                    x2 = np.random.randint(a[i],a[i+1]-1)
                R.record["age_dist_stages"][i] = range(np.random.randint(
                    min(x1,x2),max(x1,x2)))
                minl = min(minl,len(R.record["age_dist_stages"][i]))
            print "age_dist_stages[-1]:\n", R.record["age_dist_stages"][-1]
            R.n_stage = R.record["age_dist_stages"][-2][-1]
            R.population.generations[:] = R.conf["age_dist_generations"][-1][0]-1
            R.n_snap_ad = nsnap-2
            R.n_snap_ad_bool = False
            R.execute()
            minl = min(minl, len(R.record["age_dist_stages"][-1]))
            assert not R.dieoff
            assert R.record["n_snapshots"] == R.n_snap == nsnap
            assert R.record["age_distribution"].shape == (nsnap,
                    minl,maxls)
        elif R.conf["setup"] == "auto-dieoff":
            # dieoff between second and third snapshot
            # for last snapshot check age_dist_stages updated
            for i in xrange(2):
                R.record["snapshot_pops"][i] = R.population.clone()
            a = np.linspace(0,R.conf["max_stages"]/2,nsnap).astype(int)
            minl = R.conf["max_stages"]/2
            x1 = x2 = 0
            while x1==x2:
                x1 = np.random.randint(a[0],a[1]-1)
                x2 = np.random.randint(a[0],a[1]-1)
            R.record["age_dist_stages"][0] = range(np.random.randint(
                min(x1,x2),max(x1,x2)))
            minl = min(minl,len(R.record["age_dist_stages"][0]))
            R.n_stage = R.record["age_dist_stages"][0][-1]
            R.population.generations[:] = R.conf["age_dist_generations"][1][0]-1
            R.n_snap_ad = 0
            R.n_snap_ad_bool = False
            R.execute()
            minl = min(minl, len(R.record["age_dist_stages"][-1]))
            assert R.dieoff
            assert R.record["n_snapshots"] == R.n_snap == 2
            assert R.record["age_distribution"].shape == (2,
                    minl,maxls)
            # check dieoff info recording
            assert R.record["dieoff_at"].shape == (R.conf["max_fail"],2)
            assert np.all(R.record["dieoff_at"] != 0)
            assert np.allclose(R.record["dieoff_at"][:,1], R.conf["kill_at"],\
                    atol=5)
        elif R.conf["setup"] == "auto-max_stages":
            # n_stage exceeds max_stages
            R.execute()
            assert R.n_stage >= R.conf["max_stages"]
        assert R.complete
        assert R.record["finalised"]
        # save record for Plotter test
        outfile = open(os.path.join(os.path.abspath("."), "aegis/Core/Test",\
                R.conf["setup"]+".rec"),"w")
        pickle.dump(R.record, outfile)
        outfile.close()

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
