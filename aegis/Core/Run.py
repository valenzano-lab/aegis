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
from .functions import init_ages, init_genomes, init_generations
from .functions import init_gentimes, deep_eq
from .functions import timenow, timediff, get_runtime
from .Config import Config
from .Population import Population
from .Record import Record

## CLASS ##
class Run:
    def __init__(self, config, startpop, n_run, report_n, verbose):
        # Init Run parameters
        self.log = ""
        self.conf = config.copy()
        self.s_range = self.conf["s_range"]
        self.r_range = self.conf["r_range"]
        self.n_stage = 0
        self.n_snap = 0
        # special snapshot counter for age distribution
        self.n_snap_ad = 0
        self.n_snap_ad_bool = False
        self.n_run = n_run
        self.dieoff = False
        self.current_gen = 0
        self.complete = False
        self.report_n = report_n
        self.verbose = verbose
        self.resources = self.conf["res_start"]
        self.conf["prng"].shuffle(self.conf["genmap"])
        self.genmap = self.conf["genmap"] # Not a copy
        # Init Population
        if startpop != "": # If given a seeding population
            self.population = startpop.clone()
            # Adopt from population: genmap, n_base, chr_len
            self.conf["genmap"] = self.population.genmap
            self.conf["chr_len"] = self.population.chr_len
            self.conf["n_base"] = self.population.n_base
            self.population.object_max_age += self.conf["object_max_age"]
            self.conf["object_max_age"] = self.population.object_max_age
            # Keep new max_ls, maturity, sexual, g_dist, offsets, prng
            self.population.repr_mode = self.conf["repr_mode"]
            self.population.maturity = self.conf["maturity"]
            self.population.g_dist = self.conf["g_dist"]
            self.population.repr_offset = self.conf["repr_offset"]
            self.population.neut_offset = self.conf["neut_offset"]
            self.population.prng = self.conf["prng"]
            self.conf["params"] = self.conf.make_params()
        else:
            self.population = Population(self.conf["params"],self.conf["genmap"],
                self.conf["mapping"], init_ages(), init_genomes(),
                init_generations(), init_gentimes())
        # Init Record
        self.record = Record(self.conf)

    def update_resources(self):
        """If resources are variable, update them based on current
        population."""
        new_res = self.conf["res_function"](
                self.population.N, self.resources)
        self.resources = new_res

    def starving(self):
        """Determine whether population is starving based on resource level."""
        return self.conf["stv_function"](self.population.N, self.resources)

    def force_dieoff(self):
        """Determine whether to force populaton dieoff if so set in config."""
        stg = gen = False
        if self.conf["kill_at"] > 0:
            if self.conf["auto"]:
                gen = (np.min(self.population.generations) >= \
                        self.conf["kill_at"])
            elif not self.conf["auto"]:
                stg = (self.n_stage >= self.conf["kill_at"])
        return stg or gen

    def update_starvation_factors(self):
        """Update starvation factors under starvation."""
        if not self.starving() or not self.conf["pen_cuml"]:
            self.s_range = self.conf["s_range"]
            self.r_range = self.conf["r_range"]
        if self.starving():
            self.s_range = self.conf["surv_pen_func"](self.s_range,\
                        self.population.N, self.resources)
            self.r_range = self.conf["repr_pen_func"](self.r_range,\
                        self.population.N, self.resources)

    def execute_stage(self):
        """Perform one stage of a simulation run and test for completion."""
        full_report =  self.is_full_report()
        if not self.dieoff:
            # Update ages, resources and starvation
            self.population.increment_ages()
            self.update_resources()
            if full_report:
                self.logprint("Resources = {}".format(self.resources))
            self.update_starvation_factors()
            # Reproduction and death
            if full_report:
                self.logprint("Calculating reproduction and death...")
            n0 = self.population.N
            self.population.growth(self.r_range, self.conf["m_rate"],
                    self.conf["m_ratio"],
                    self.conf["r_rate"])
            n1 = self.population.N
            if self.force_dieoff(): self.s_range[:] = 0
            self.population.death(self.s_range)
            n2 = self.population.N
            if full_report:
                self.logprint("Done. {0} individuals born, {1} died."\
                        .format(n1-n0,n1-n2))
        # Record
        self.record_stage(full_report)
        # Update run status
        self.dieoff = self.record["dieoff"] = (self.population.N == 0)
        self.n_stage += 1
        self.test_complete()
        if self.complete and not self.dieoff:
            self.record.finalise()

    def record_stage(self, full_report=False):
        """Record and report population information, as appropriate for
        the stage number and run settings."""
        # Set up reporting parameters
        report_stage = (self.n_stage % self.report_n == 0)
        if report_stage:
            s = "Population = {0}.".format(self.population.N)
            if self.conf["auto"] and self.population.N > 0:
                g = " Min generation = {0}/{1}."
                s += g.format(np.min(self.population.generations),
                        self.conf["min_gen"])
            self.logprint(s)
        if self.population.N == 0:
            self.dieoff = True
            return
        # Decide whether to take a detailed snapshot
        snapshot = -1
        if not self.conf["auto"]:
            if self.n_stage in self.conf["snapshot_stages"]:
                snapshot = self.n_snap
        else:
            obs = np.min(self.population.generations)
            exp = self.conf["snapshot_generations_remaining"][0]
            if obs >= exp:
                snapshot = self.n_snap
                # Save at which stages are the snapshots taken
                self.record["snapshot_stages"][snapshot] = self.n_stage
                # Prevent same min generation triggering multiple snapshots:
                self.conf["snapshot_generations_remaining"] = \
                        self.conf["snapshot_generations_remaining"][1:]
        # Decide whether to record age_distribution
        age_dist_rec = -1
        if self.conf["age_dist_N"] == "all":
            age_dist_rec = self.n_stage
        elif not self.conf["auto"]:
            if self.n_stage in self.conf["age_dist_stages"]:
                age_dist_rec = self.n_stage
        else:
            if obs in self.conf["age_dist_generations"]:
                age_dist_rec = self.n_stage
                if self.n_snap_ad_bool:
                    self.n_snap_ad += 1
                    self.n_snap_ad_bool = False
                # Save at which stages age_dist is recorded
                self.record["age_dist_stages"][self.n_snap_ad].append(self.n_stage)
            else:
                self.n_snap_ad_bool = True
        # Record information and return verbosity boolean
        self.record.update(self.population, self.resources, self.starving(),
                self.n_stage, snapshot, age_dist_rec)
        self.n_snap += 1 if snapshot >= 0 else 0
        if (snapshot >= 0) and full_report: self.logprint("Snapshot taken.")

    def is_full_report(self):
        report_stage = (self.n_stage % self.report_n == 0)
        return report_stage and self.verbose

    def test_complete(self):
        """Test whether a run is complete following a given stage,
        under fixed and automatic stage counting."""
        if not self.dieoff and self.conf["auto"]:
            self.current_gen = np.min(self.population.generations)
            gen = (self.current_gen >= self.conf["min_gen"])
            stg = (self.n_stage >= self.conf["max_stages"])
        elif not self.dieoff and not self.conf["auto"]:
            stg, gen = (self.n_stage >= self.conf["n_stages"]), False
        self.complete = self.dieoff or gen or stg

    def execute_attempt(self):
        """Execute a single run attempt from start to completion or failure."""
        # Compute starting time and announce run start
        if not hasattr(self, "starttime"): self.starttime = timenow(False)
        f,r = self.record["prev_failed"]+1, self.n_run
        a = "run {0}, attempt {1}".format(r,f) if f>1 else "run {0}".format(r)
        self.logprint("Beginning {0} at {1}.".format(a, timenow(True)))
        if self.conf["auto"]:
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

    def execute(self):
        """Execute the run, repeating until either an attempt is
        successful or the maximum number of failures is reached."""
        prev_failed = self.record["prev_failed"]
        if self.conf["max_fail"] > 1: save_state = self.copy()
        self.execute_attempt()
        if self.dieoff:
            nfail =  prev_failed + 1
            self.logprint("Run failed. Total failures = {}.".format(nfail))
            # Record dieoff info
            if self.conf["auto"]:
                self.record["dieoff_at"][nfail-1] = [self.n_stage,self.current_gen]
            else:
                self.record["dieoff_at"][nfail-1] = self.n_stage
            if nfail >= self.conf["max_fail"]: # Accept failure and terminate
                self.logprint("Failure limit reached. Accepting failed run.")
                self.record.finalise()
                self.logprint(get_runtime(self.starttime, self.endtime))
            else: # Reset to saved state (except for log, prev_failed and dieoff at)
                save_state.record["prev_failed"] = nfail
                save_state.record["dieoff_at"] = self.record["dieoff_at"]
                save_state.log = self.log + "\n"
                attrs = vars(save_state)
                # Revert everything else (except for prng)
                for key in attrs:
                    setattr(self, key, attrs[key])
                self.conf = save_state.conf
                self.population = save_state.population
                self.record = save_state.record
                return self.execute()
        self.logprint(get_runtime(self.starttime, self.endtime))

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        # Compute numbers of spaces to keep all messages aligned
        n, r = self.conf["n_stages"], self.conf["n_runs"]
        if n == "auto": n = self.conf["max_stages"]
        nspace_run = len(str(r-1))-len(str(self.n_run))
        nspace_stg = len(str(n)) - len(str(self.n_stage))
        # Create string
        lstr = "RUN {0}{1} | STAGE {2}{3} | {4}".format(" "*nspace_run,
                self.n_run, " "*nspace_stg, self.n_stage, message)
        print lstr
        self.log += lstr+"\n"

    # Basic methods
    def copy(self):
        self_copy = copy.deepcopy(self)
        self_copy.conf = self.conf.copy()
        self_copy.population = self.population.clone()
        self_copy.record = self.record.copy()
        return self_copy

    # Comparison methods

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return NotImplemented
        variables = ["log", "n_stage", "n_snap", "n_run",\
                "dieoff", "complete", "report_n", "verbose", "resources"]
        for k in variables:
            if getattr(self, k) != getattr(other, k): return False
        if not np.array_equal(self.genmap, other.genmap): return False
        if not np.array_equal(self.s_range, other.s_range): return False
        if not np.array_equal(self.r_range, other.r_range): return False
        conf_same = (self.conf == other.conf)
        pop_same = (self.population == other.population)
        rec_same = (self.record == other.record)
        return conf_same and pop_same and rec_same

    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(vars(self))
