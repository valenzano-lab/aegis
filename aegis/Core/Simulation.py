########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Simulation                                                  #
# Description: Wrapper object representing a complete simulation call, #
#   including seeding, run execution and output.                       #
########################################################################

## PACKAGE IMPORT ##
import numpy as np
import scipy.stats as st
import copy, datetime, time, os, importlib, inspect, shutil
from .functions import init_ages, init_genomes, init_generations
from .functions import timenow, timediff, get_runtime, init_gentimes
from .Config import Config
from .Population import Population
from .Record import Record
from .Run import Run
try:
       import cPickle as pickle
except:
       import pickle

## CLASS ##
class Simulation:
    """An object representing a single complete simulation."""

    # INITIALISATION

    def __init__(self, config_file, report_n, verbose, outpath=""):
        """Initialise the simulation and report starting information."""
        self.starttime, self.log = timenow(False), ""
        self.logprint("\nBeginning simulation {}.".format(timenow()))
        self.logprint("Working directory: "+os.getcwd())
        self.get_conf(config_file)
        self.conf.generate()
        self.report_n, self.verbose = report_n, verbose
        self.outpath = os.getcwd() if outpath=="" else outpath
        self.get_startpop(self.conf["path_to_seed_file"])
        self.init_runs()

    def get_conf(self, file_name):
        """Import specified configuration file for simulation."""
        try:
            self.conf = Config(file_name)
        except ImportError:
            s = "No such file in simulation directory: " + file_name
            self.abort(ImportError, s)

    def get_seed(self, seed_path):
        """Import a population seed from a pickled AEGIS object."""
        try:
            infile = open(seed_path, "rb")
            obj = pickle.load(infile)
            if not isinstance(obj, Population):
                s = "Seed import failed: {} does not hold a Population\
                        object.".format(seed_path)
                self.abort(TypeError, s)
            infile.close()
            return obj
        except IOError:
            s = "Seed import failed: no file or directory under {}".format(seed_path)
            self.abort(IOError, s)

    def get_seed_all(self, seed_path):
        """Import N = number of runs populations from given seed path."""
        self.logprint("Reading seed population from ./{}".format(seed_path))
        if seed_path.endswith(".pop"):
            self.logprint("Import succeeded.")
            return [self.get_seed(seed_path)]
        elif os.path.isdir(seed_path):
            pop_files = [f for f in os.listdir(seed_path) if f.endswith(".pop")]
            if len(pop_files) != self.conf["n_runs"]:
                s = "Number of seed files does not equal to number of runs.\nTried getting seeds from: ./{}.".format(seed_path)
                self.abort(ImportError, s)
            pop_files.sort()
            seeds = [self.get_seed(os.path.join(seed_path,l)) for l in pop_files]
            self.logprint("Import succeeded.")
            return seeds
        # Otherwise abort
        s = "Seed path must point to a *.pop file or a directory containing *.pop files."
        self.abort(ImportError, s)

    def get_startpop(self, seed=""):
        """Import a population seed from a pickled AEGIS object, or
        return blank to generate a new starting population."""
        # Return a blank string list if no seed given
        if seed == "":
            self.logprint("Seed: None.")
            self.startpop = [""]
            return
        # Otherwise, get list of seed populations
        pops = self.get_seed_all(seed)
        self.startpop = pops

    def init_runs(self):
        self.logprint("Initialising runs...")
        y,x = (len(self.startpop) == 1), xrange(self.conf["n_runs"])
        self.runs = [Run(self.conf, self.startpop[0 if y else n], n,
            self.report_n, self.verbose) for n in x]
        self.logprint("Runs initialised.\n")

    # LOGGING, SAVING & ABORTING

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        print message
        self.log += message+"\n"

    def logsave(self):
        """Finalise and save Simulation log to file."""
        log_file = open(self.conf["output_prefix"] + "_log.txt", "w")
        try:
            log_file.write(self.log)
        finally:
            log_file.close()

    def abort(self, errtype, message):
        """Print an error message to the Simulation log, then abort."""
        self.log += "\n{0}: {1}\n".format(errtype.__name__, message)
        self.endtime = timenow(False)
        self.log += "\nSimulation terminated {}.".format(timenow())
        self.logsave()
        raise errtype(message)

    # EXECUTION

    def execute_series(self):
        """Execute simulation runs in series, with no external
        parallelisation."""
        for n in xrange(self.conf["n_runs"]):
            self.runs[n].execute()

    def execute(self):
        """Execute simulation runs."""
        self.execute_series()

    # FINALISATION

    def save_output(self):
        """Save output data according to the specified output mode."""
        # Auxiliary functions
        def intro(otype, suffix):
            self.logprint("Saving {}...".format(otype))
            dirname = os.path.join(self.outpath,\
                    self.conf["output_prefix"] + "_files/{}".format(suffix))
            if os.path.exists(dirname): # Overwrite existing output
                    shutil.rmtree(dirname)
            os.makedirs(dirname)
            return(dirname)
        def save(obj, filename):
            try:
                f = open(filename, "wb")
                pickle.dump(obj, f)
            finally:
                f.close()
        def outro(otype): self.logprint("{} saved.".format(otype).capitalize())
        # Saving output
        if self.conf["output_mode"] >= 2: # Save all snapshot pops
            dirname = intro("snapshot populations", "populations/snapshots")
            for n in xrange(self.conf["n_runs"]):
                for m in xrange(self.conf["n_snapshots"]):
                    pop = self.runs[n].record["snapshot_pops"][m]
                    filename = dirname + "/run{0}_s{1}.pop".format(n,m)
                    save(pop, filename)
                del self.runs[n].record["snapshot_pops"]
                outro("snapshot populations")
        if self.conf["output_mode"] >= 1: # Save final populations
            dirname = intro("final populations", "populations/final")
            for n in xrange(self.conf["n_runs"]):
                pop = self.runs[n].record["final_pop"]
                filename = dirname + "/run{}.pop".format(n)
                save(pop, filename)
                del self.runs[n].record["final_pop"]
            outro("final populations")
        if self.conf["output_mode"] >= 0: # Save records
            dirname = intro("run records", "records")
            for n in xrange(self.conf["n_runs"]):
                rec = self.runs[n].record
                filename = dirname + "/run{}.rec".format(n)
                save(rec, filename)
            outro("run records")

    def finalise(self):
        # Log simulation completion
        self.endtime = timenow(False)
        self.logprint("\nSimulation completed at {}.".format(timenow(True)))
        self.logprint(get_runtime(self.starttime, self.endtime,
            "Total runtime (excluding output)"))
        # Perform and log output
        outstart = timenow(False)
        self.logprint("\nBeginning output at {}.".format(timenow(True)))
        self.save_output()
        outend = timenow(False)
        self.logprint("Output complete at {}.".format(timenow(True)))
        self.logprint(get_runtime(outstart, outend, "Time for output"))
        # Finish and terminate
        self.logprint(get_runtime(self.starttime, outend,
            "Total runtime (including output)"))
        self.logprint("Exiting.\n")

    # OTHER

    def copy(self):
        self_copy = copy.deepcopy(self)
        self_copy.conf = self.conf.copy()
        for i in range(self.conf["n_runs"]):
            self_copy.runs[i] = self.runs[i].copy()
        return self_copy

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return NotImplemented
        variables = ["starttime", "log", "report_n", "verbose"]
        for k in variables:
            if getattr(self,k) != getattr(other,k): return False
        if self.startpop != other.startpop: return False
        for i in range(self.conf["n_runs"]):
            if self.runs[i] != self.runs[i]: return False
        return True

    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented
