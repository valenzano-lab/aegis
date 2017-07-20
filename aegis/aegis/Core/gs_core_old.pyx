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
