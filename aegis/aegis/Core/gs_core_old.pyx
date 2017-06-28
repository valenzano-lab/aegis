# cython: profile=True

# Modules
import numpy as np
cimport numpy as np
import scipy.stats as st
import importlib, operator, time, os, random, datetime, copy, multiprocessing
import sys
try:
       import cPickle as pickle
except:
       import pickle

# Cython data types for numpy
NPINT = np.int # Integer arrays
NPFLOAT = np.float # Decimal arrays
NPBOOL = np.uint8 # Boolean arrays

ctypedef np.int_t NPINT_t
ctypedef np.float_t NPFLOAT_t
ctypedef np.uint8_t NPBOOL_t

#############
# FUNCTIONS #
#############

def get_runtime(starttime, endtime):
    """Convert two datetime.now() outputs into a time difference 
    in human-readable units."""
    runtime = endtime - starttime
    days = runtime.days
    hours = runtime.seconds/3600
    minutes = runtime.seconds/60 - hours*60
    seconds = runtime.seconds - minutes*60 - hours*3600
    delta = "Total runtime: "
    if days != 0: delta += "{d} days, ".format(d=days)
    if hours != 0: delta += "{h} hours, ".format(h=hours)
    if minutes != 0: delta += "{m} minutes, ".format(m=minutes)
    delta += "{s} seconds.".format(s=seconds)
    return delta

def execute_run(run, maxfail):
    """Execute a simulation run, handling and recording failed runs
    as appropriate."""
    if maxfail>0: blank_run = copy.deepcopy(run)
    run.execute()
    if maxfail>0 and run.dieoff:
        nfail = blank_run.record.get("prev_failed")
        if nfail >= maxfail - 1:
            run.logprint("Total failures = {0}.".format(nfail+1))
            run.logprint("Failure limit reached. Accepting failed run.")
            run.logprint(get_runtime(run.starttime, run.endtime))
            return run
        nfail += 1
        run.logprint("Run failed. Total failures = {0}. Repeating..."\
                .format(nfail))
        blank_run.record.set("prev_failed", nfail)
        def divplus1(x): return (x*100)/(x+1)
        blank_run.record.set("percent_dieoff",
                divplus1(blank_run.record.get("prev_failed")))
        blank_run.log = run.log + "\n"
        blank_run.starttime = run.starttime
        return execute_run(blank_run, maxfail)
    run.logprint(get_runtime(run.starttime, run.endtime))
    return run

class Run:
    """An object representing a single run of a simulation."""
    def __init__(self, config, startpop, n_run, report_n, verbose):
        self.log = ""
        self.conf = copy.deepcopy(config)
        self.surv_penf = 1.0
        self.repr_penf = 1.0
        self.resources = self.conf.res_start
        np.random.shuffle(self.conf.genmap)
        self.genmap = self.conf.genmap # Not a copy
        if startpop != "":
            self.population = startpop.clone()
            # Adopt from population: genmap, nbase, chrlen
            self.conf.genmap = self.population.genmap
            self.conf.chr_len = self.population.chrlen
            self.conf.n_base = self.population.nbase
            # Keep new max_ls, maturity, sexual
            self.population.maturity = self.conf.maturity
            self.population.maxls = self.conf.max_ls
            self.population.sex = self.conf.sexual
            self.conf.params = {"sexual":self.conf.sexual,
                    "chr_len":self.conf.chr_len, "n_base":self.conf.n_base,
                    "maturity":self.conf.maturity, "max_ls":self.conf.max_ls,
                    "age_random":self.conf.age_random,
                    "start_pop":self.conf.start_pop, "g_dist":self.conf.g_dist}
        else:
            self.population = Outpop(Population(self.conf.params,
                self.conf.genmap, testage(), testgen()))
        self.n_stage = 0
        self.n_snap = 0
        self.n_run = n_run
        self.record = Record(self.conf)
        self.dieoff = False
        self.complete = False
        self.report_n = report_n
        self.verbose = verbose

    def update_resources(self):
        """If resources are variable, update them based on current 
        population."""
        if self.conf.res_var: # Else do nothing
            k = 1 if self.population.N > self.resources else self.conf.V
            new_res = int((self.resources-self.population.N)*k + self.conf.R)
            self.resources = np.clip(new_res, 0, self.conf.res_limit)

    def starving(self):
        """Determine whether population is starving based on resource level."""
        if self.conf.res_var:
            return self.resources == 0
        else:
            return self.population.N > self.resources

    def update_starvation_factors(self):
        """Update starvation factors under starvation."""
        if self.starving():
            self.surv_penf *= self.conf.death_inc if self.conf.surv_pen else 1.0
            self.repr_penf *= self.conf.repr_dec if self.conf.repr_pen else 1.0
        else: 
            self.surv_penf = 1.0
            self.repr_penf = 1.0

    def execute_stage(self):
        """Perform one stage of a simulation run and test for completion."""
        if not isinstance(self.population, Population):
            m="Convert Outpop objects to Population before running execute_stage."
            raise TypeError(m)
        report_stage = (self.n_stage % self.report_n == 0)
        if report_stage:
            self.logprint("Population = {0}.".format(self.population.N))
        self.dieoff = (self.population.N == 0)
        if not self.dieoff:
            # Record information
            snapshot = -1 if self.n_stage not in self.conf.snapshot_stages \
                    else self.n_snap
            full_report = report_stage and self.verbose
            self.record.update(self.population, self.resources, self.surv_penf,
                    self.repr_penf, self.n_stage, snapshot)
            self.n_snap += (1 if snapshot >= 0 else 0)
            if (snapshot >= 0) and full_report: self.logprint("Snapshot taken.")
            # Update ages, resources and starvation
            self.population.increment_ages()
            self.update_resources()
            self.update_starvation_factors()
            if full_report: self.logprint(
                    "Starvation factors = {0} (survival), {1} (reproduction)."\
                            .format(self.surv_penf,self.repr_penf))
            # Reproduction and death
            if full_report: self.logprint("Calculating reproduction and death...")
            n0 = self.population.N
            self.population.growth(self.conf.r_range, self.repr_penf,
                    self.conf.r_rate, self.conf.m_rate, self.conf.m_ratio)
            n1 = self.population.N
            self.population.death(1-self.conf.s_range, self.surv_penf)
            n2 = self.population.N
            if full_report: 
                self.logprint("Done. {0} individuals born, {1} died."\
                        .format(n1-n0,n1-n2))
            if self.n_stage in self.conf.crisis_stages or chance(self.conf.crisis_p):
                self.population.crisis(self.conf.crisis_sv)
                self.logprint("Crisis! {0} individuals died, {1} survived."\
                        .format(n2-self.population.N, self.population.N))
        # Update run status
        self.dieoff = self.record.record["dieoff"] = (self.population.N == 0)
        self.record.record["percent_dieoff"] = self.dieoff*100.0
        self.n_stage += 1
        self.complete = self.dieoff or self.n_stage==self.conf.number_of_stages
        if self.complete and not self.dieoff:
            self.record.finalise()

    def execute(self):
        """Execute a run object from start to completion."""
        self.population = self.population.toPop()
        if not hasattr(self, "starttime"): 
            self.starttime = datetime.datetime.now()
        runstart = time.strftime('%X %x', time.localtime())
        f,r = self.record.record["prev_failed"]+1, self.n_run
        a = "run {0}, attempt {1}".format(r,f) if f>1 else "run {0}".format(r)
        self.logprint("Beginning {0} at {1}.".format(a, runstart))
        while not self.complete:
            self.execute_stage()
        self.population = Outpop(self.population)
        self.endtime = datetime.datetime.now()
        runend = time.strftime('%X %x', time.localtime())
        b = "Extinction" if self.dieoff else "Completion"
        self.logprint("{0} at {1}. Final population: {2}"\
                .format(b, runend, self.population.N))
        if f>0 and not self.dieoff: 
            self.logprint("Total attempts required: {0}.".format(f))

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        # Compute numbers of spaces to keep all messages aligned
        nspace_run = len(str(self.conf.number_of_runs-1))-\
                len(str(self.n_run))
        nspace_stg = len(str(self.conf.number_of_stages-1))\
                -len(str(self.n_stage))
        # Create string
        lstr = "RUN {0}{1} | STAGE {2}{3} | {4}".format(" "*nspace_run, 
                self.n_run, " "*nspace_stg, self.n_stage, message)
        print lstr
        self.log += lstr+"\n"

class Simulation:
    """An object representing a simulation, as defined by a config file."""

    def __init__(self, config_file, seed, seed_n, report_n, verbose):
        self.starttime = datetime.datetime.now()
        simstart = time.strftime('%X %x', time.localtime())+".\n"
        self.log = ""
        self.logprint("\nBeginning simulation at "+simstart)
        self.logprint("Working directory: "+os.getcwd())
        self.get_conf(config_file)
        self.conf.generate()
        self.get_startpop(seed, seed_n)
        self.report_n, self.verbose = report_n, verbose
        self.logprint("Initialising runs...")
        y,x = (len(self.startpop) == 1), xrange(self.conf.number_of_runs)
        self.runs = [Run(self.conf, self.startpop[0 if y else n], n,
            self.report_n, self.verbose) for n in x]
        self.logprint("Runs initialised. Executing...\n")

    def execute(self, nproc=-1, maxfail=10):
        """Execute all runs."""
        if nproc <= 0: # Use all available cores
            pool = multiprocessing.Pool()
        elif nproc == 1: # Run without multiprocessing
            self.runs = [execute_run(r, maxfail) for r in self.runs]
            return
        else: # Use specifed number of cores
            pool = multiprocessing.Pool(nproc)
        lock = multiprocessing.Lock()
        lock.acquire()
        try:
            asyncruns = []
            for n in xrange(self.conf.number_of_runs):
                asyncruns+= [pool.apply_async(execute_run, [self.runs[n],
                    maxfail])]
            outruns = [x.get() for x in asyncruns]
            self.runs = outruns
            self.log += "\n".join([x.log for x in self.runs])
        finally:
            lock.release()

    def get_conf(self, file_name):
        """Import specified configuration file for simulation."""
        try:
            conf = importlib.import_module(file_name)
        except ImportError:
            print "No such file in simulation directory: " + file_name
            q = raw_input(
                    "Enter correct config file name, or skip to abort: ")
            if q == "":
                exit("Aborting: no valid configuration file given.")
            else:
                return self.get_conf(q)
        self.conf = Config(conf)

    def get_startpop(self, seed="", pop_number=-1):
        """Import any seed simulation (or return blank)."""
        if seed == "":
            self.logprint("Seed: None.")
            self.startpop = [""]
            return
        p = "all populations" if pop_number==-1 else "population "+str(pop_number)
        if isinstance(seed, Simulation):
            status = "Seeding {0} directly from Simulation object."
            simobj = seed
        else:
            try:
                # Make sure includes extension (default "sim")
                seed += ".sim" if os.path.splitext(seed)[1] == "" else ""
                status = "Seed: "+seed+", {0}."
                simfile = open(seed, "rb")
                simobj = pickle.load(simfile) # import simulation object
            except IOError:
                print "No such seed file: " + seed
                q = raw_input(
                        "Enter correct path to seed file, or skip to abort: ")
                if q == "": exit("Aborting.")
                r = raw_input("Enter population seed number, or enter -1 to \
                        seed all populations, or skip to abort.")
                if r == "": exit("Aborting.")
                return self.get_startpop(q, r)
        nruns = len(simobj.runs)
        if pop_number == -1:
            # -1 = seed all populations to equivalent runs in new sim
            if nruns != self.conf.number_of_runs:
                message = "Number of runs in seed file ({0}) ".format(nruns)
                message += "does not match current configuration ({0} runs)."\
                        .format(self.conf.number_of_runs)
                raise IndexError(message)
            self.logprint(status.format("all populations"))
            self.startpop = [r.population for r in simobj.runs]
        else:
            if pop_number >= nruns:
                print "Population seed number out of range. Possible \
                        values: 0 to {0}".format(nruns-1)
                q = raw_input("Enter a new population number, or enter -1 \
                        to seed all populations, or skip to abort.")
                while q not in range(-1, nruns): 
                    # Repeat until valid input given
                    if q == "": exit("Aborting.")
                    q = raw_input("Please enter a valid integer from -1 \
                            to {0}, or skip to abort.".format(nruns-1))
                return self.get_startpop(seed, q)
            self.logprint(status.format("population {0}".format(pop_number)))
            self.startpop = [simobj.runs[pop_number].population]
        return

    def finalise(self, file_pref, log_pref):
        """Finish recording and save output files."""
        self.average_records()
        self.endtime = datetime.datetime.now()
        simend = time.strftime('%X %x', time.localtime())+"."
        self.logprint("\nSimulation completed at "+simend)
        self.logprint(get_runtime(self.starttime, self.endtime))
        self.logprint("Saving output and exiting.\n")
        sim_file = open(file_pref + ".sim", "wb")
        log_file = open(log_pref + ".txt", "w")
        rec_file = open(file_pref + ".rec", "wb")
        try:
            log_file.write(self.log)
            pickle.dump(self, sim_file)
            if hasattr(self, "avg_record"):
                pickle.dump(self.avg_record, rec_file)
            else:
                pickle.dump(self.runs[0].record, rec_file)
        finally:
            sim_file.close()
            log_file.close()
            rec_file.close()

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        print message
        self.log += message+"\n"

    def average_records(self):
        if len(self.runs) == 1: 
            self.avg_record = self.runs[0].record
            return
        self.avg_record = Record(self.conf)
        rec_list = [x.record for x in self.runs if x.complete and not x.dieoff]
        rec_gets = [r.get for r in rec_list] # Get record get methods
        # Auxiliary functions
        def test_compatibility():
            """Test that all records to be averaged have compatible dimensions
            in all data entries."""
            eq_array_0 = np.array([[len(r("genmap")), r("chr_len"), r("n_base"),
                r("max_ls"), r("maturity"), r("n_states"), r("n_neutral"),
                r("surv_step"), r("repr_step")] for r in rec_gets])
            eq_array_1 = np.array([list(r("surv_bound"))+list(r("repr_bound"))+\
                    list(r("snapshot_stages")) for r in rec_gets])
            eq_array_2 = np.array([r("mean_gt")["s"].shape+r("density")["s"].shape+\
                    r("entropy_gt")["s"].shape+r("resources").shape+r("n1").shape+\
                    r("age_distribution").shape for r in rec_gets])
            cm = np.all(np.isclose(eq_array_0, eq_array_0[0])) and \
                    np.all(np.isclose(eq_array_1, eq_array_1[0])) and \
                    np.all(np.isclose(eq_array_2,eq_array_2[0]))
            if not cm:
                np.set_printoptions(threshold=np.inf)
                print eq_array_0-eq_array_0[0]
                print eq_array_1-eq_array_1[0]
                print eq_array_2-eq_array_2[0]
                raise ValueError("Cannot generate average run data;"+\
                        " runs incompatible.")
            else:
                self.logprint("Runs are compatible; generating averaged data.")
        def average_entry(key):
            """Average a given entry across all runs and store under the
            corresponding key in self.avg_record."""
            if key in ["snapshot_pops", "n_runs", "n_successes"]: return
            k0,sar = rec_gets[0](key), self.avg_record
            if key in ["population_size","resources","surv_penf","repr_penf",
                    "prev_failed", "percent_dieoff",
                    "population_size_window_mean", "resources_window_mean",
                    "population_size_window_var", "resources_window_var"]:
                # Concatenate rather than average
                sar.set(key,np.vstack([r(key) for r in rec_gets]))
            elif isinstance(k0, dict):
                d_out, d_out_sd = {}, {}
                for k in sorted(k0.keys()):
                    karray = np.array([r(key)[k] for r in rec_gets])
                    d_out[k] = np.nanmean(karray, 0) 
                    d_out_sd[k] = np.nanstd(karray, 0)
                    # nanmean/nanstd to account for actual death rate nan's
                sar.set(key, d_out)
                sar.set(key + "_sd", d_out_sd)
            elif isinstance(k0, np.ndarray) or isinstance(k0, int)\
                    or isinstance(k0, float):
                karray = np.array([r(key) for r in rec_gets])
                sar.set(key, np.nanmean(karray, 0))
                sar.set(key + "_sd", np.nanstd(karray, 0))
                if isinstance(sar.get(key),np.ndarray):
                    if np.allclose(sar.get(key), sar.get(key).astype(int)):
                            sar.set("key", sar.get(key).astype(int))
                elif sar.get(key) == int(sar.get(key)):
                    sar.set(key,int(sar.get(key)))
            else:
                erstr = "Unrecognised entry type: {0}, {1}".format(k0,type(k0))
                raise ValueError(erstr)
        def compute_failure():
            """Explicitly calculate number and percentage of failed runs."""
            sar = self.avg_record
            sar.set("n_runs",len(self.runs))
            sar.set("n_successes", 
                    sum([x.complete and not x.dieoff for x in self.runs]))
            fails = [r.record.get("prev_failed") for r in self.runs]
            failsum = np.sum(fails)
            sar.set("total_failed", failsum)
            sar.set("percent_dieoff_total", 100*\
                    (sar.get("total_failed")+sum([x.dieoff for x in self.runs]))/\
                    (sar.get("total_failed")+len(self.runs)))
        # Procedure
        test_compatibility() # First test record compatibility
        keys = rec_list[0].get_keys()
        for key in keys: average_entry(key)
        compute_failure()
