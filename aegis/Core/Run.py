########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Run                                                         #
# Description: Class representing a single, independent simulation     #
#   run, including initialisation, execution, and output.              #
########################################################################

## PACKAGE IMPORT ##
import numpy as np
import scipy.stats as st
import copy, datetime, time
from .functions import chance, init_ages, init_genomes, init_generations
from .functions import init_gentimes
from .functions import timenow, timediff, get_runtime
from .Config import Infodict, Config, deepeq
from .Population import Population
from .Record import Record

## CLASS ##
class Run:
    def __init__(self, config, startpop, n_run, report_n, verbose):
        # init Run parameters
        self.log = ""
        self.conf = config.copy()
        self.surv_penf = 1.0
        self.repr_penf = 1.0
        self.n_stage = 0
        self.n_snap = 0
        self.n_run = n_run
        self.dieoff = False
        self.complete = False
        self.report_n = report_n
        self.verbose = verbose
        self.resources = self.conf["res_start"]
        np.random.shuffle(self.conf["genmap"])
        self.genmap = self.conf["genmap"] # Not a copy
        # init Population
        #n = self.conf["number_of_stages"] if not self.auto() else self.conf["max_stages"]
        if startpop != "": # If given a seeding population
            self.population = startpop.clone()
            # Adopt from population: genmap, n_base, chr_len
            self.conf["genmap"] = self.population.genmap
            self.conf["chr_len"] = self.population.chr_len
            self.conf["n_base"] = self.population.n_base
            self.population.object_max_age += self.conf["object_max_age"]
            self.conf["object_max_age"] = self.population.object_max_age
            # Keep new max_ls, maturity, sexual, g_dist, offsets
            self.population.repr_mode = self.conf["repr_mode"]
            self.population.maturity = self.conf["maturity"]
            self.population.g_dist = self.conf["g_dist"]
            self.population.repr_offset = self.conf["repr_offset"]
            self.population.neut_offset = self.conf["neut_offset"]
            self.conf["params"] = self.conf.subdict(
                    ["repr_mode", "chr_len", "n_base", "maturity", "start_pop",
                        "max_ls", "g_dist", "repr_offset", "neut_offset", "object_max_age"])
        else:
            self.population = Population(self.conf["params"],
                self.conf["genmap"], init_ages(), init_genomes(),
                init_generations(), init_gentimes())
        # init Record
        self.record = Record(self.conf)

    def update_resources(self):
        """If resources are variable, update them based on current
        population."""
        if self.conf["res_var"]: # Else do nothing
            k = 1 if self.population.N > self.resources else self.conf["V"]
            new_res = int((self.resources-self.population.N)*k + self.conf["R"])
            self.resources = np.clip(new_res, 0, self.conf["res_limit"])

    # NOTE WARNING: this is conceptually different from the old method since
    # now resources can never be lesser than self.conf["R"]? what do you mean by
    # checking "if r < 0" when r is a user-defined constant?
    #
    # if you mean that this method should be written in a way that we can do:
    # def starving: return self.population.N > self.resources
    # than I agree, but I think this as it is is wrong
    def update_resources_new(self):
        """Simpler, more intuitive resource updating."""
        if self.conf["res_var"]: # Else do nothing
            v,r,l = self.conf["V"], self.conf["R"], self.conf["res_limit"]
            new_res = max(0, self.resources - self.N) * v + r
            self.resources = new_res if r < 0 else min(r, new_res)
        #! TODO: Test and implement this

    def starving(self):
        """Determine whether population is starving based on resource level."""
        if self.conf["res_var"]:
            return self.resources == 0
        else:
            return self.population.N > self.resources

    def update_starvation_factors(self):
        """Update starvation factors under starvation."""
        if self.starving():
            if self.conf["surv_pen"]: self.surv_penf *= self.conf["death_inc"]
            if self.conf["repr_pen"]: self.repr_penf *= self.conf["repr_dec"]
        else:
            self.surv_penf = 1.0
            self.repr_penf = 1.0

    def execute_stage(self):
        """Perform one stage of a simulation run and test for completion."""
        # Make sure population is in cythonised form
        full_report =  self.record_stage()
        if not self.dieoff:
            # Update ages, resources and starvation
            self.population.increment_ages()
            self.update_resources()
            self.update_starvation_factors()
            if full_report: self.logprint(
                    "Starvation factors = {0} (survival), {1} (reproduction)."\
                            .format(self.surv_penf,self.repr_penf))
            # Reproduction and death
            if full_report:
                self.logprint("Calculating reproduction and death...")
            n0 = self.population.N
            self.population.growth(self.conf["r_range"], self.repr_penf,
                    self.conf["m_rate"], self.conf["m_ratio"],
                    self.conf["r_rate"])
            n1 = self.population.N
            self.population.death(self.conf["s_range"], self.surv_penf)
            n2 = self.population.N
            if full_report:
                self.logprint("Done. {0} individuals born, {1} died."\
                        .format(n1-n0,n1-n2))
        # Update run status
        self.dieoff = self.record["dieoff"] = (self.population.N == 0)
        self.n_stage += 1
        self.test_complete()
        if self.complete and not self.dieoff:
            self.record.finalise()
        #! TODO: What about if dieoff?

    def record_stage(self):
        """Record and report population information, as appropriate for
        the stage number and run settings."""
        # Set up reporting parameters
        report_stage = (self.n_stage % self.report_n == 0)
        if report_stage:
            s = "Population = {0}.".format(self.population.N)
            if self.conf.auto():
                g = " Min generation = {0}/{1}."
                s += g.format(np.min(self.population.generations),
                        self.conf["min_gen"])
            self.logprint(s)
        if self.population.N == 0:
            self.dieoff = True
            return
        # Decide whether to take a detailed snapshot
        snapshot = -1
        if not self.conf.auto():
            if self.n_stage in self.conf["snapshot_stages"]:
                snapshot = self.n_snap
        else:
            obs = np.min(self.population.generations)
            exp = self.conf["snapshot_generations_remaining"][0]
            if obs >= exp:
                snapshot = self.n_snap
                # Prevent same min generation triggering multiple snapshots:
                self.conf["snapshot_generations_remaining"] = \
                        self.conf["snapshot_generations_remaining"][1:]
                        # TODO: Reconstruct this in Record finalisation
        # Record information and return verbosity boolean
        self.record.update(self.population, self.resources, self.surv_penf,
                self.repr_penf, self.n_stage, snapshot)
        self.n_snap += 1 if snapshot >= 0 else 0
        full_report = report_stage and self.verbose
        if (snapshot >= 0) and full_report: self.logprint("Snapshot taken.")
        return full_report

    def test_complete(self):
        """Test whether a run is complete following a given stage,
        under fixed and automatic stage counting."""
        if not self.dieoff and self.conf.auto():
            gen = (np.min(self.population.generations) >= self.conf["min_gen"])
            stg = (self.n_stage >= self.conf["max_stages"])
        elif not self.dieoff and not self.conf.auto():
            stg, gen = (self.n_stage >= self.conf["number_of_stages"]), False
        self.complete = self.dieoff or gen or stg
        # TODO: Test this

    def execute_attempt(self):
        """Execute a single run attempt from start to completion or failure."""
        # Compute starting time and announce run start
        if not hasattr(self, "starttime"): self.starttime = timenow(False)
        f,r = self.record["prev_failed"]+1, self.n_run
        a = "run {0}, attempt {1}".format(r,f) if f>1 else "run {0}".format(r)
        self.logprint("Beginning {0} {1}.".format(a, timenow(True)))
        if self.conf.auto():
            self.logprint("Automatic stage counting. Target generation: {}."\
                    .format(self.conf["min_gen"]))
        # Execute stages until completion
        while not self.complete:
            self.execute_stage()
        # Compute end time and announce run end
        self.endtime = timenow(False)
        b = "Extinction" if self.dieoff else "Completion"
        self.logprint("{0} {1}. Final population: {2}"\
                .format(b, timenow(True), self.population.N))
        if f>0 and not self.dieoff:
            self.logprint("Total attempts required: {0}.".format(f))
        # return self.record ?

    def execute(self):
        """Execute the run, repeating until either an attempt is
        successful or the maximum number of failures is reached."""
        if self.conf["max_fail"] > 1: save_state = self.copy()
        self.execute_attempt()
        if self.dieoff:
            nfail = save_state.record["prev_failed"] + 1
            self.logprint("Run failed. Total failures = {}.".format(nfail))
            if nfail >= self.conf["max_fail"]: # Accept failure and terminate
                self.logprint("Failure limit reached. Accepting failed run.")
                self.logprint(get_runtime(self.starttime, self.endtime))
            else: # Reset to saved state (except for log and prev_failed)
                save_state.record["prev_failed"] = nfail
                save_state.log = self.log + "\n"
                attrs = vars(save_state)
                for key in attrs: # Revert everything else
                    setattr(self, key, attrs[key])
                #! TODO: Test that this does not change start time
                return self.execute()
        self.logprint(get_runtime(self.starttime, self.endtime))

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        # Compute numbers of spaces to keep all messages aligned
        n, r = self.conf["number_of_stages"], self.conf["number_of_runs"]
        if n == "auto": n = self.conf["max_stages"] # TODO: test this
        nspace_run = len(str(r-1))-len(str(self.n_run))
        nspace_stg = len(str(n)) - len(str(self.n_stage))
        # Create string
        lstr = "RUN {0}{1} | STAGE {2}{3} | {4}".format(" "*nspace_run,
                self.n_run, " "*nspace_stg, self.n_stage, message)
        print lstr
        self.log += lstr+"\n"

    # Basic methods
    def copy(self):
        return copy.deepcopy(self)

    # Comparison methods

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return NotImplemented
        return deepeq(vars(self), vars(other))
        #for a in ["log", "conf", "surv_penf", "other_penf", "resources",
        #        "genmap", "population", "n_stage", "n_snap", "n_run"
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(vars(self))
