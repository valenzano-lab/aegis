#!/usr/bin/python

##########################
## CONFIGURE SIMULATION ##
##########################

print "\nImporting libraries and config...",
from sys import argv
from random import sample
import time
import numpy as np
import cPickle
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
    x = 1
    resources = c.res_start
    n_snap = 0 # number of first snapshot

    ## generating starting population
    print "Generating starting population...",
    # First row of population array
    population = []
    # Rest of population
    for i in range(c.start_pop-1):
        indiv = fn.make_individual(c.age_random, c.variance,
                c.chr_len, c.gen_map, c.s_dist, c.r_dist)
        population.append(indiv)
    population = np.array(population)
    print "done."

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
        
        # Get (proportional) age distribution:
        n_age = np.bincount(population[:,0])/N
#        n_age_txt.append(n_age)
        if n_stage in c.snapshot_stages:
            record = fn.update_record(record, population, N, resources, 
                    c.gen_map, c.chr_len, c.n_base, c.d_range, c.r_range,
                    c.maturity, c.max_ls, c.window_size, n_stage, n_snap)
            n_snap += 1
        else:
            fn.quick_update(record, n_stage, N, resources)

        # everyone gets 1 stage older
        population[:,0] += 1
        ages = population[:,0]

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

    ## pop-pyramid output
    males_females_age = n_age*100
#    males_females_age_txt = list(males_females_age)

    ## text file
#    txt_file =  open('plot_values_run'+str(n_run)+'.txt','wb')
##    cPickle.dump(pop_txt,txt_file)
#    cPickle.dump(res_txt,txt_file)
#    cPickle.dump(n_age_txt,txt_file)
#    cPickle.dump(repr_rate_txt,txt_file)
#    cPickle.dump(repr_rate_sd_txt,txt_file)
#    cPickle.dump(repr_rate_junk_txt,txt_file)
#    cPickle.dump(surv_rate_txt,txt_file)
#    cPickle.dump(surv_rate_sd_txt,txt_file)
#    cPickle.dump(surv_rate_junk_txt,txt_file)
#    cPickle.dump(repr_fit_txt,txt_file)
#    cPickle.dump(repr_fit_junk_txt,txt_file)
#    cPickle.dump(surv_fit_txt,txt_file)
#    cPickle.dump(surv_fit_junk_txt,txt_file)
#    cPickle.dump(density_surv_txt,txt_file)
#    cPickle.dump(density_repr_txt,txt_file)
#    cPickle.dump(hetrz_mea_txt,txt_file)
#    cPickle.dump(hetrz_mea_sd_txt,txt_file)
#    cPickle.dump(males_females_age_txt,txt_file)
#    txt_file.close()

#    pop_file = open('pop_'+str(n_stage)+'_run'+str(n_run)+'.txt','wb')
    cPickle.dump(population,pop_file)
    cPickle.dump(resources,pop_file)
    if c.res_var:
        cPickle.dump(c.R,pop_file)
        cPickle.dump(c.V,pop_file)
        cPickle.dump(c.res_limit,pop_file)
    cPickle.dump(c.death_bound[1],pop_file)
    cPickle.dump(c.repr_bound[1],pop_file)
    cPickle.dump(c.r_rate,pop_file)
    cPickle.dump(c.m_rate,pop_file)
    cPickle.dump(c.gen_map,pop_file)
    pop_file.close()
