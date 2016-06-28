#!/usr/bin/python

# Define arguments and enable help
import argparse,os,time
parser = argparse.ArgumentParser(description='Run the genome ageing \
        simulation.')
parser.add_argument('-s', metavar="<str>", default="",
        help="path to population seed file (default: no seed)")
parser.add_argument('-c', metavar='<str>', default="config",
        help="name of configuration file within simulation directory \
                (default: config.py)")
parser.add_argument('dir', help="path to simulation directory")
parser.add_argument('-r', type=int, metavar="<int>", default=10,
        help="report information every <int> stages (default: 10)")
parser.add_argument('-P', '--profile', action="store_true",
        help="profile genome simulation with cProfile")
parser.add_argument('-v', '--verbose', action="store_true",
        help="display full information at each report stage \
                (default: only starting population)")
args = parser.parse_args()
print args

# Import other libraries
import pyximport; pyximport.install()
from random import sample
import numpy as np
import gs_functions as fn
from gs_classes import Population,Record
from datetime import datetime
if args.profile:
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()

simstart = datetime.now()
simstart_print = time.strftime('%X %x', time.localtime())+".\n"
if args.profile: pr.enable() # start profiling

###################################
## PARSE ARGUMENTS AND CONFIGURE ##
###################################

fn.get_dir(args.dir) # Change to simulation directory
fn.logprint("\nBeginning simulation at "+simstart_print)
fn.logprint("Working directory: "+os.getcwd())

c = fn.get_conf(args.c) # Import config file as "c".
startpop = fn.get_startpop(args.s) # Get seed population, if any.
genmap = np.copy(c.genmap)
np.random.shuffle(genmap) # Randomize order of genetic units.

####################
## RUN SIMULATION ##
####################

for n_run in range(1, c.number_of_runs+1):
    runstart = datetime.now()
    runstart_print = time.strftime('%X %x', time.localtime())+".\n"
    fn.logprint("\nBeginning run "+str(n_run)+" at ", False)
    fn.logprint(runstart_print)

    # # # # # # # # # # # # # # #
    # 1: INITIALISE POPULATION  #
    # # # # # # # # # # # # # # #

    ## Generate starting population (if no seed)
    if startpop != "": population = startpop
    else:
        fn.logprint("Generating starting population...", False)
        population = Population(c.params, genmap)
        fn.logprint("done.")

    ## Initialise record
    record = Record(population, c.snapshot_stages,
            c.number_of_stages, c.d_range, c.r_range, c.window_size)
    n_snap = 0 # number of previous snapshots

    surv_penf = 1.0 # Neutral survival penalty factor (increases under starvation)
    repr_penf = 1.0 # Neutral reproduction penalty factor (same as above)
    resources = c.res_start

    # # # # # # # # #
    # 2: STAGE LOOP #
    # # # # # # # # #

    fn.logprint("Beginning stage loop.")
    for n_stage in range(c.number_of_stages):
        # Determine whether to print full stage information:
        full_report = n_stage%args.r == 0 and args.verbose
        # Print stage report:
        if(population.N==0):
            fn.logprint("\nPerished at stage "+str(n_stage+1)+".")
            break
        elif n_stage%args.r == 0:
            fn.logprint ("\nStage "+str(n_stage+1)+": ", False)
            fn.logprint (str(population.N)+" individuals.")

        # Record output variables
        if n_stage in c.snapshot_stages:
            if full_report: fn.logprint("Taking snapshot...",False)
            record.update(population, resources, surv_penf, repr_penf, n_stage, n_snap)
            n_snap += 1
            if full_report: fn.logprint("done.")
        else:
            record.quick_update(n_stage, population, resources, surv_penf, repr_penf)

        population.increment_ages()

        # Change in resources and penalty under starvation
        if c.res_var: # variable; function of population size
            resources = fn.update_resources(resources, population.N, c.R,
                    c.V, c.res_limit, full_report)
            surv_penf = surv_penf*c.death_inc if c.surv_pen and resources == 0 else 1.0
            repr_penf = repr_penf*c.repr_dec if c.repr_pen and resources == 0 else 1.0
        else: # constant; penalty increases if population exceeds resources
            surv_penf = surv_penf*c.death_inc if c.surv_pen and population.N>resources else 1.0
            repr_penf = repr_penf*c.repr_dec if c.repr_pen and population.N>resources else 1.0
        if full_report: fn.logprint("Survival starvation factor: "+str(surv_penf)+"\n"+\
                                    "Reproduction starvation factor"+str(repr_penf))

        # Reproduction & death
        population.growth(c.r_range, repr_penf, c.r_rate, c.m_rate, c.m_ratio,
                full_report)

        population.death(c.d_range, surv_penf, full_report)

        if n_stage in c.crisis_stages:
            population.crisis(c.crisis_sv, n_stage)

    ## RUN ENDED
    runend = datetime.now()
    runend_print = time.strftime('%X %x', time.localtime())+". "
    fn.logprint("\nEnd of run "+str(n_run)+" at ", False)
    fn.logprint(runend_print+"Final population: "+str(population.N)+".")
    fn.print_runtime(runstart, runend)

    ## WRITE POPULATION, RECORD TO FILE ##
    fn.run_output(n_run, population, record, c.window_size)

simend = datetime.now()
simend_print = time.strftime('%X %x', time.localtime())+"."
fn.logprint("\nSimulation completed at "+simend_print)
fn.print_runtime(simstart, simend)
fn.logprint("Exiting.")
if args.profile:
    pr.create_stats()
    pr.dump_stats('timestats.txt')
