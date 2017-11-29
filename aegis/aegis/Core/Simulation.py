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
import copy, datetime, time, os, importlib, inspect, numbers, shutil
from .functions import chance, init_ages, init_genomes, init_generations
from .functions import timenow, timediff, get_runtime
from .Config import Infodict, Config
from .Population import Population, Outpop
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

    def __init__(self, config_file, report_n, verbose):
        """Initialise the simulation and report starting information."""
        self.starttime, self.log = timenow(False), ""
        self.logprint("\nBeginning simulation {}.".format(timenow()))
        self.logprint("Working directory: "+os.getcwd())
        self.get_conf(config_file)
        self.conf.generate()
        # NOTE what does this do? "cong" should be "conf"
        if isinstance(self.conf["random_seed"], numbers.Number):
            random.seed(self.conf["random_seed"])
        self.report_n, self.verbose = report_n, verbose
        #self.get_startpop(self.conf["path_to_seed_file"], self.conf["seed_run_n"])
        self.get_startpop(self.conf["path_to_seed_file"])
        self.init_runs()

    def get_conf(self, file_name):
        """Import specified configuration file for simulation."""
        try:
            conf = importlib.import_module(file_name)
            self.conf = Config(conf)
        except ImportError:
            s = "No such file in simulation directory: " + file_name
            self.abort(ImportError, s)

    def get_seed(self, seed_path):
        """Import a population seed from a pickled AEGIS object."""
        try:
            self.logprint("Reading seed population from {}".format(seed_path))
            infile = open(seed_path, "rb")
            obj = pickle.load(infile)
            self.logprint("Import succeeded. Seed type = {}".format(type(obj)))
            return obj
        except IOError:
            s = "Seed import failed: no such file or directory."
            self.abort(IOError, s)
        finally:
            infile.close()

    def get_startpop(self, seed="", pop_number=-1):
        """Import a population seed from a pickled AEGIS object, or
        return blank to generate a new starting population."""
        # Return a blank string list if no seed given
        if seed == "":
            self.logprint("Seed: None.")
            self.startpop = [""]
            return
        # Otherwise, import seed object and act based on type
        obj = self.get_seed(seed)
        # Obtain Population object based on type of imported object
        self.set_startpop(obj, pop_number)

    def set_startpop(self, obj, pop_number=-1):
        """Set self.startpop from an AEGIS object based on its type."""
        # Dispatch to __startpop__ method of input object
        if hasattr(obj, "__startpop__"):
            pop, msg = obj.__startpop__(pop_number)
            if inspect.isclass(pop) and issubclass(pop, Exception):
                self.abort(pop, msg)
            else:
                self.logprint(msg)
                self.startpop = pop
        else:
            msg = "No method for extracting seed from object type: {}"
            msg = msg.format(type(obj))
            self.abort(ValueError, msg)

        # TODO: Require pop_number to be an integer.

    def init_runs(self):
        self.logprint("Initialising runs...")
        y,x = (len(self.startpop) == 1), xrange(self.conf["number_of_runs"])
        self.runs = [Run(self.conf, self.startpop[0 if y else n], n,
            self.report_n, self.verbose) for n in x]
        self.logprint("Runs initialised.\n")

    # __startpop__ METHOD

    def __startpop__(self, pop_number):
        if pop_number < 0:
            msg = "No method for obtaining seed '-1' from Simulation object."
            pop = ValueError
        else:
            pop, msg = self.runs[pop_number].__startpop__(-1)
            msg = "Setting seed from embedded Run of Simulation object ({})."
            msg = msg.format(pop_number)
        return (pop, msg)

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
        self.log += "\nSimulation terminated{}.".format(timenow())
        self.logsave()
        raise errtype(message)

    # EXECUTION

    def execute_series(self):
        """Execute simulation runs in series, with no external
        parallelisation."""
        for n in xrange(self.conf["number_of_runs"]):
            self.runs[n].execute()

    def execute(self):
        """Execute simulation runs."""
        self.execute_series()

    # FINALISATION

    def save_output(self):
        """Save output data according to the specified output mode."""
        #! TODO: Higher output levels (save runs, save sim)?
        # Auxiliary functions
        def intro(otype, suffix):
            self.logprint("Saving {}...".format(otype))
            dirname = self.conf["output_prefix"] + "_output/{}".format(suffix)
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
            for n in xrange(self.conf["number_of_runs"]):
                for m in xrange(self.conf["number_of_snapshots"]):
                    pop = self.runs[n].record["snapshot_pops"][m]
                    filename = dirname + "/run{0}_s{1}.pop".format(n,m)
                    #! TODO: Correct file name for number of digits
                    save(pop, filename)
                del self.runs[n].record["snapshot_pops"]
                outro("snapshot populations")
        if self.conf["output_mode"] >= 1: # Save final populations
            dirname = intro("final populations", "populations/final")
            for n in xrange(self.conf["number_of_runs"]):
                pop = self.runs[n].record["final_pop"]
                filename = dirname + "/run{}.pop".format(n)
                save(pop, filename)
                del self.runs[n].record["final_pop"] #! TODO: Test that this works
            outro("final populations")
        if self.conf["output_mode"] >= 0: # Save records
            dirname = intro("run records", "records")
            for n in xrange(self.conf["number_of_runs"]):
                rec = self.runs[n].record
                filename = dirname + "/run{}.rec".format(n)
                save(rec, filename)
            outro("run records")

    def finalise(self):
        # self.average_records() #! TODO: Decide what to do about averaging
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

    def copy(self): return copy.deepcopy(self)
