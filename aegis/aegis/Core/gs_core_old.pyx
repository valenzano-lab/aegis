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

class Simulation:
    """An object representing a simulation, as defined by a config file."""

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
