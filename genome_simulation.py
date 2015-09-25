#!/usr/bin/python

from random import sample
import numpy as np
import gs_functions as fn
import argparse,os,time
from datetime import datetime

simstart = datetime.now()
simstart_print = time.strftime('%X %x', time.localtime())+".\n"

###################################
## PARSE ARGUMENTS AND CONFIGURE ##
###################################

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
parser.add_argument('-v', '--verbose', action="store_true",
        help="display full information at each report stage \
                (default: only starting population)")

args = parser.parse_args()
print args
fn.get_dir(args.dir) # Change to simulation directory
log = open("log.txt", "w")
fn.logprint("\nBeginning simulation at "+simstart_print, log)
fn.logprint("Working directory: "+os.getcwd(), log)

c = fn.get_conf(args.c, log) # Import config file as "c".
startpop = fn.get_startpop(args.s, log) # Get seed population, if any.
gen_map = np.copy(c.gen_map)
np.random.shuffle(gen_map)

####################
## RUN SIMULATION ##
####################

for n_run in range(1, c.number_of_runs+1):
    runstart = datetime.now()
    runstart_print = time.strftime('%X %x', time.localtime())+".\n"
    fn.logprint("\nBeginning run "+str(n_run)+" at ", log, False)
    fn.logprint(runstart_print, log)

    # # # # # # # # # # # # # # #
    # 1: INITIALISE POPULATION  #
    # # # # # # # # # # # # # # # 
    
    record = fn.initialise_record(c.snapshot_stages, 
            c.number_of_stages, c.max_ls, c.n_base, gen_map, c.chr_len, 
            c.d_range, c.r_range, c.window_size)
    x = 1.0 # Initial starvation factor
    resources = c.res_start
    n_snap = 0 # number of previous snapshots

    ## Generate starting population (if no seed)
    population = startpop if startpop != "" else fn.make_population(
            c.start_pop, c.age_random, c.max_ls, c.maturity, c.variance, 
            c.n_base, c.chr_len, gen_map, c.s_dist, c.r_dist, log)

    # # # # # # # # #
    # 2: STAGE LOOP #
    # # # # # # # # # 

    fn.logprint("Beginning stage loop.", log)
    for n_stage in range(c.number_of_stages):

        N = len(population)
        # Determine whether to print full stage information:
        full_report = n_stage%args.r == 0 and args.verbose
        # Print stage report:
        if(N==0):
            fn.logprint("\nPerished at stage "+str(n_stage+1)+".", log)
            break
        elif n_stage%args.r == 0:
            fn.logprint ("\nStage "+str(n_stage+1)+": ", log, False)
            fn.logprint (str(N)+" individuals.", log)
        
        # Record output variables
        if n_stage in c.snapshot_stages:
            if full_report: fn.logprint("Taking snapshot...",log,False)
            record = fn.update_record(record, population, N, resources, 
                    x, gen_map, c.chr_len, c.n_base, c.d_range, 
                    c.r_range, c.maturity, c.max_ls, n_stage, n_snap)
            n_snap += 1
            if full_report: fn.logprint("done.",log)
        else:
            fn.quick_update(record, n_stage, N, resources, x)

        population[:,0] += 1 # everyone gets 1 stage older

        # Change in resources and starvation
        if c.res_var: # function of population
            resources = fn.update_resources(resources, N, c.R, c.V, 
                    c.res_limit, log, full_report)
            x = x*c.death_inc if resources == 0 else 1.0
        else: # constant; death rate increases if population exceeds
            x = x*c.death_inc if N>resources else 1.0
        if full_report: fn.logprint("Starvation factor: "+str(x), log)

        # Reproduction
        population = fn.reproduction(population, c.maturity, c.max_ls, 
                gen_map, c.n_base, c.chr_len, c.r_range, c.m_rate, 
                c.m_ratio, c.r_rate, log, c.sexual, full_report)

        # Death
        population = fn.death(population, c.max_ls, gen_map, c.n_base, 
                c.chr_len, c.d_range, x, log, full_report)

        # Extrinsic death crisis:
        if n_stage in c.crisis_stages:
            N = len(population)
            n_survivors = int(N*c.crisis_sv)
            population = population[sample(range(N), n_survivors)]
            fn.logprint("Stage " + n_stage + ": Crisis! ", log, False)
            fn.logprint(str(n.survivors)+" individuals survived.", log)

    ## RUN ENDED
    runend = datetime.now()
    runend_print = time.strftime('%X %x', time.localtime())+". "
    fn.logprint("\nEnd of run "+str(n_run)+" at ", log, False)
    fn.logprint(runend_print+"Final population: "+str(N)+".", log)
    fn.print_runtime(runstart, runend, log)

    ## WRITE POPULATION, RECORD TO FILE ##
    fn.run_output(n_run, population, record, log, c.window_size)

simend = datetime.now()
simend_print = time.strftime('%X %x', time.localtime())+"."
fn.logprint("\nSimulation completed at "+simend_print, log)
fn.print_runtime(simstart, simend, log)
fn.logprint("Exiting.", log)
