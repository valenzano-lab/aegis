#!/usr/bin/python

from random import sample
import numpy as np
import functions as fn
import argparse

###################################
## PARSE ARGUMENTS AND CONFIGURE ##
###################################

parser = argparse.ArgumentParser(description='Run the genome ageing simulation.')
parser.add_argument('-s', metavar="<str>", default="",
        help="path to population seed file (default: no seed)")
parser.add_argument('-c', metavar='<str>', default="config",
        help="name of configuration file within simulation directory (default: config.py)")
parser.add_argument('dir', help="path to simulation directory")
parser.add_argument('-r', type=int, metavar="<int>", default=10,
        help="report information every <int> stages (default: 10)")
parser.add_argument('-v', '--verbose', action="store_true",
        help="display full information at each report stage (default: only starting population)")

args = parser.parse_args()
print args
fn.get_dir(args.dir)
c = fn.get_conf(args.c) # Import config file as "c".
startpop = fn.get_startpop(args.s) # Get seed population, if any.
gen_map = np.copy(c.gen_map)

####################
## RUN SIMULATION ##
####################

for n_run in range(1, c.number_of_runs+1):

    # # # # # # # # # # # # # # #
    # 1: INITIALISE POPULATION  #
    # # # # # # # # # # # # # # # 
    
    print "\nBeginning run "+str(n_run)+".\n"
    np.random.shuffle(gen_map) # reshuffle genome every run
    record = fn.initialise_record(c.snapshot_stages, 
            c.number_of_stages, c.max_ls, c.n_base, gen_map, c.chr_len, 
            c.d_range, c.r_range, c.window_size)
    x = 1.0 # Initial starvation factor
    resources = c.res_start
    n_snap = 0 # number of first snapshot

    ## Generate starting population (if no seed)
    population = startpop if startpop != "" else fn.make_population(
            c.start_pop, c.age_random, c.variance, c.chr_len, 
            gen_map, c.s_dist, c.r_dist)

    # # # # # # # # #
    # 2: STAGE LOOP #
    # # # # # # # # # 

    print "Beginning stage loop."
    for n_stage in range(c.number_of_stages):

        N = len(population)
        full_report = n_stage%args.r == 0 and args.verbose

        if(N==0):
            print "\nPerished at stage "+str(n_stage+1)+"."
            break
        elif n_stage%args.r == 0:
            print "\nStage "+str(n_stage+1)+": "+str(N)+" individuals."
        
        # Record output variables
        if n_stage in c.snapshot_stages:
            if full_report: print "Taking snapshot...",
            record = fn.update_record(record, population, N, resources, x, 
                    gen_map, c.chr_len, c.n_base, c.d_range, c.r_range,
                    c.maturity, c.max_ls, c.window_size, n_stage, n_snap)
            n_snap += 1
            if full_report: print "done."
        else:
            fn.quick_update(record, n_stage, N, resources, x)

        population[:,0] += 1 # everyone gets 1 stage older

        # Change in resources
        if c.res_var: # function of population
            resources = fn.update_resources(resources, N, c.R, c.V, 
                    c.res_limit, full_report)
            x = x*c.death_inc if resources == 0 else 1.0
        else: # constant; death rate increases if population exceeds
            x = x*c.death_inc if N>resources else 1.0
        if full_report: print "Starvation factor: "+str(x)

        # Reproduction
        population = fn.reproduction(population, c.maturity, 
                c.max_ls, gen_map, c.chr_len, c.r_range, c.m_rate, 
                c.r_rate, c.sexual, full_report)
        N = len(population)

        # Death
        population = fn.death(population, c.max_ls, gen_map, c.chr_len,
                c.d_range, x, full_report)

        # Extrinsic death crisis:
        if n_stage in c.crisis_stages:
            N = len(population)
            n_survivors = int(N*c.crisis_sv)
            population = population[sample(range(N), n_survivors)]
            print "Stage " + n_stage + ": Crisis! " + str(n.survivors),
            print " individuals survived."

    ## RUN ENDED
    print "\nEnd of run "+str(n_run)+". Final population: "+str(N)+".\n"

    ## WRITE POPULATION, RECORD TO FILE ##
    fn.run_output(n_run, population, record)
