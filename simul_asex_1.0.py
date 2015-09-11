#!/usr/bin/python

##########################
## CONFIGURE SIMULATION ##
##########################

print "\nImporting libraries and config...",
from sys import argv
from random import sample
import numpy as np
import simul_functions as fn

fn.getwd(argv)
c = fn.getconf(argv) # Import config file as "c".
print "done.\n"

####################
## RUN SIMULATION ##
####################

for n_run in range(1, c.number_of_runs+1):

    # # # # # # # # # # # # # # #
    # 1: INITIALISE POPULATION  #
    # # # # # # # # # # # # # # # 
    
    print "\nBeginning run "+str(n_run)+".\n"
    np.random.shuffle(c.gen_map) # reshuffle genome every run
    record = fn.initialise_record(c.number_of_snapshots, 
            c.number_of_stages, c.max_ls, c.n_base, c.chr_len, 
            c.window_size)
    x = 1.0 # Initial starvation factor
    resources = c.res_start
    n_snap = 0 # number of first snapshot

    ## Generate starting population
    population = fn.make_population(c.start_pop, c.age_random, 
            c.variance, c.chr_len, c.gen_map, c.s_dist, c.r_dist)

    # # # # # # # # #
    # 2: STAGE LOOP #
    # # # # # # # # # 

    print "Beginning stage loop."
    for n_stage in range(c.number_of_stages):

        N = len(population)
        
        if(N==0):
            print "\nPerished at stage "+str(n_stage+1)+"."
            break
        elif c.verbose or n_stage%10==0:
            print "\nStage "+str(n_stage+1)+": "+str(N)+" individuals."
        
        # Record output variables
        if n_stage in c.snapshot_stages:
            record = fn.update_record(record, population, N, resources, x, 
                    c.gen_map, c.chr_len, c.n_base, c.d_range, c.r_range,
                    c.maturity, c.max_ls, c.window_size, n_stage, n_snap)
            n_snap += 1
        else:
            fn.quick_update(record, n_stage, N, resources, x)

        population[:,0] += 1 # everyone gets 1 stage older

        # Change in resources
        if c.res_var: # function of population
            resources = fn.update_resources(resources, N, c.R, c.V, 
                    c.res_limit, c.verbose)
            x = x*c.death_inc if resources == 0 else 1.0
        else: # constant; death rate increases if population exceeds
            x = x*c.death_inc if N>resources else 1.0
        if c.verbose: print "Starvation factor: "+str(x)

        # Reproduction
        population = fn.reproduction_asex(population, c.maturity, c.max_ls,
                c.gen_map, c.chr_len, c.r_range, c.m_rate, c.verbose)
        N = len(population)

        # Death
        population = fn.death(population, c.max_ls, c.gen_map, c.chr_len,
                c.d_range, x, c.verbose)

        # Extrinsic death crisis:
        if n_stage in c.crisis_stages:
            N = len(population)
            n_survivors = int(N*c.crisis_sv)
            population = population[sample(range(N), n_survivors)]
            if c.verbose:
                print "Crisis! "+str(n.survivors)+" individuals survived."

    ## RUN ENDED
    print "\nEnd of run "+str(n_run)+".\n"

    ## WRITE POPULATION, RECORD TO FILE ##
    fn.run_output(n_run, population, record)
