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
import copy
from .functions import chance, init_ages, init_genomes, init_generations
from .Config import Infodict
from .Population import Population, Outpop
from .Record import Record

## CLASS ##
class Run:
    def __init__(self, config, startpop, n_run, report_n, verbose):
        self.log = ""
        self.conf = config.copy()
        self.surv_penf = 1.0
        self.repr_penf = 1.0
        self.resources = self.conf["res_start"]
        np.random.shuffle(self.conf["genmap"])
        self.genmap = self.conf["genmap"] # Not a copy
        if startpop != "": # If given a seeding population
            self.population = startpop.clone()
            # Adopt from population: genmap, n_base, chr_len
            self.conf["genmap"] = self.population.genmap
            self.conf["chr_len"] = self.population.chr_len
            self.conf["n_base"] = self.population.n_base
            # Keep new max_ls, maturity, sexual, g_dist, offsets
            self.population.repr_mode = self.conf["repr_mode"]
            self.population.maturity = self.conf["maturity"]
            self.population.g_dist = self.conf["g_dist"]
            self.population.repr_offset = self.conf["repr_offset"]
            self.population.neut_offset = self.conf["neut_offset"]
            self.conf["params"] = self.conf["subdict"](
                    ["repr_mode", "chr_len", "n_base", "maturity", "start_pop",
                        "max_ls", "g_dist", "repr_offset", "neut_offset"])
        else:
            self.population = Outpop(Population(self.conf["params"],
                self.conf["genmap"], init_ages(), init_genomes(), 
                init_generations()))
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
        if self.conf["res_var"]: # Else do nothing
            k = 1 if self.population.N > self.resources else self.conf["V"]
            new_res = int((self.resources-self.population.N)*k + self.conf["R"])
            self.resources = np.clip(new_res, 0, self.conf["res_limit"])

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
        if not isinstance(self.population, Population):
            m="Convert Outpop objects to Population before running execute_stage."
            raise TypeError(m)
        # Set up reporting parameters
        report_stage = (self.n_stage % self.report_n == 0)
        if report_stage:
            self.logprint("Population = {0}.".format(self.population.N))
        self.dieoff = (self.population.N == 0)
        if not self.dieoff:
            # Record information
            snapshot = -1 if self.n_stage not in self.conf["snapshot_stages"] \
                    else self.n_snap
            full_report = report_stage and self.verbose
            self.record.update(self.population, self.resources, self.surv_penf,
                    self.repr_penf, self.n_stage, snapshot)
            self.n_snap += 1 if snapshot >= 0 else 0
            if (snapshot >= 0) and full_report: self.logprint("Snapshot taken.")
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
        self.record["percent_dieoff"] = self.dieoff*100.0
        self.n_stage += 1
        self.complete = self.dieoff or self.n_stage==self.conf["number_of_stages"]
        if self.complete and not self.dieoff:
            print "---"
            self.record.finalise()
        #! TODO: What about if dieoff?

    def execute(self):
        """Execute a run object from start to completion."""
        self.population = self.population.toPop() # Convert from Outpop
        # Compute starting time and announce run start
        if not hasattr(self, "starttime"): 
            self.starttime = datetime.datetime.now()
        runstart = time.strftime('%X %x', time.localtime())
        f,r = self.record["prev_failed"]+1, self.n_run
        a = "run {0}, attempt {1}".format(r,f) if f>1 else "run {0}".format(r)
        self.logprint("Beginning {0} at {1}.".format(a, runstart))
        # Execute stages until completion
        while not self.complete:
            self.execute_stage()
        self.population = Outpop(self.population) # Convert back to Outpop
        # Compute end time and announce run end
        self.endtime = datetime.datetime.now()
        runend = time.strftime('%X %x', time.localtime())
        b = "Extinction" if self.dieoff else "Completion"
        self.logprint("{0} at {1}. Final population: {2}"\
                .format(b, runend, self.population.N))
        if f>0 and not self.dieoff: 
            self.logprint("Total attempts required: {0}.".format(f))
        # return self.record ?

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        # Compute numbers of spaces to keep all messages aligned
        nspace_run = len(str(self.conf["number_of_runs"]-1))-\
                len(str(self.n_run))
        nspace_stg = len(str(self.conf["number_of_stages"]-1))\
                -len(str(self.n_stage))
        # Create string
        lstr = "RUN {0}{1} | STAGE {2}{3} | {4}".format(" "*nspace_run, 
                self.n_run, " "*nspace_stg, self.n_stage, message)
        print lstr
        self.log += lstr+"\n"

    # Basic methods
    def copy(self):
        return copy.deepcopy(self)

