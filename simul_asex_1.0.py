#!/usr/bin/python

## CONFIGURE RUN ##

print "\nImporting libraries and config..."
from sys import argv
from random import sample
import time
import numpy as np
import cPickle
import simul_functions as fn

fn.getwd(argv)
c = fn.getconf(argv) # Import config file as "c".
print ""

# lightweight version - outputs age_distr every stage and rest only snapshot data, no sha_wie

print "Initialising...",

## DERIVE ADDITIONAL PARAMETERS ##
gen_map = np.asarray(range(0,c.max_ls)+range(c.maturity+100,
    c.max_ls+100)+[201])
# Genome map: survival (0 to max), reproduction (maturity to max), neutral
chr_len = len(gen_map)*c.n_base # Length of chromosome in binary units
d_range = np.linspace(c.death_bound[1], c.death_bound[0],2*c.n_base+1) 
# max to min death rate
r_range = np.linspace(c.repr_bound[0],c.repr_bound[1],2*c.n_base+1) 
# min to max repr rate

## plot variables
snapshot_plot_stages = np.around(np.linspace(0,c.number_of_stages,16),0)
ipx = np.arange(0,71)
surv_fit_var = np.linspace(0,1,21)
repr_fit_var = np.linspace(0,1,21)

# Write info file.
time_print =  time.strftime('%X %x', time.gmtime())

#info_file = open(out+'info.txt', 'w')
#if res_var:
#    res_comm_var = '\n\nres_var = pop\nResource increment per stage: '+str(R)+'\nResource regrowth factor: '+str(V)+'\nResource upper res_limit: '+str(res_limit)+'\nStarting amount of resources: '+str(res_start)
#else:
#    res_comm_var = '\n\nres_var = const\nResources: '+str(resources)
#info_file.write('Version 1.0\nStarted at: '+time_print+'\n\nRandom starting ages: '+str(age_random)+'\nVariance index: '+str(variance)+'\nStarting number of individuals: '+str(start_pop)+'\nNumber of stages: '+str(number_of_stages)+'\nStages of crisis occurence: '+str(crisis_stages)+'\nFraction of crisis survivors: '+str(crisis_sv)+'\nMaximal death rate: '+str(death_bound[1])+'\nInitial survival rate (distribution)[%]: '+str(s_dist)+'\nMaximal reproduction rate: '+str(repr_bound[1])+'\nRecombination rate: '+str(r_rate)+'\nMutation rate var: '+str(m_rate)+'\nNumber of runs: '+str(number_of_runs)+'\n\nComment: '+comment)
#info_file.close()

################
## SIMULATION ##
################
print "done."
for n_run in range(1, c.number_of_runs+1):
    print ""
    print "Beginning run "+str(n_run)+"."
    print ""
    np.random.shuffle(gen_map) # reshuffle genome every run
    ## constants and lists
    resources = c.res_start
    population = []
    pop_txt = []
    res_txt = []
    n_age_txt = []
    repr_rate_txt = []
    repr_rate_sd_txt = []
    repr_rate_junk_txt = []
    surv_rate_txt = []
    surv_rate_sd_txt = []
    surv_rate_junk_txt = []
    repr_fit_txt = []
    repr_fit_junk_txt = []
    surv_fit_txt = []
    surv_fit_junk_txt = []
    density_surv_txt = []
    density_repr_txt = []
    hetrz_mea_txt = []
    hetrz_mea_sd_txt = []
    x = 1

    ## generating starting population
    print "Generating starting population...",
    # First row of population array
    population = []
    # Rest of population
    for i in range(c.start_pop-1):
        indiv = fn.make_individual(c.age_random, c.variance,
                chr_len, gen_map, c.s_dist, c.r_dist)
        population.append(indiv)
    population = np.array(population)
    print "done."

    ## starting population output:
    pop_file = open('pop_0_run'+str(n_run)+'.txt','wb')
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
    cPickle.dump(gen_map,pop_file)
    pop_file.close()

    # # # # # # # # #
    # 1: STAGE LOOP #
    # # # # # # # # # 
    print "Beginning stage loop."
    for n_stage in range(0, c.number_of_stages+1):

        N = len(population)
        if(N==0):
            print "Perished at stage "+str(n_stage)+"."
            break
        elif c.verbose or n_stage%10==0:
            print "Stage "+str(n_stage)+": "+str(N)+" individuals."
        
        pop_txt.append(N)
        res_txt.append(resources)

        # Get (proportional) age distribution:
        n_age = np.bincount(population[:,0])/N
        n_age_txt.append(n_age)

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
        population = fn.reproduction_asex(population, c.maturity, c.max_ls, gen_map, chr_len, r_range, c.m_rate, c.verbose)
        N = len(population)

        # Death
        population = fn.death(population, c.max_ls, gen_map, chr_len,
                d_range, x, c.verbose)

        # Extrinsic death crisis:
        if n_stage in c.crisis_stages:
            N = len(population)
            n_survivors = int(N*c.crisis_sv)
            population = population[sample(range(N), n_survivors)]
            print "Crisis! "+str(len(population))+" individuals survived."

    ## RUN ENDED

    ## pop-pyramid output
    males_females_age = n_age*100
    males_females_age_txt = list(males_females_age)

    ## text file
    txt_file =  open('plot_values_run'+str(n_run)+'.txt','wb')
    cPickle.dump(pop_txt,txt_file)
    cPickle.dump(res_txt,txt_file)
    cPickle.dump(n_age_txt,txt_file)
    cPickle.dump(repr_rate_txt,txt_file)
    cPickle.dump(repr_rate_sd_txt,txt_file)
    cPickle.dump(repr_rate_junk_txt,txt_file)
    cPickle.dump(surv_rate_txt,txt_file)
    cPickle.dump(surv_rate_sd_txt,txt_file)
    cPickle.dump(surv_rate_junk_txt,txt_file)
    cPickle.dump(repr_fit_txt,txt_file)
    cPickle.dump(repr_fit_junk_txt,txt_file)
    cPickle.dump(surv_fit_txt,txt_file)
    cPickle.dump(surv_fit_junk_txt,txt_file)
    cPickle.dump(density_surv_txt,txt_file)
    cPickle.dump(density_repr_txt,txt_file)
    cPickle.dump(hetrz_mea_txt,txt_file)
    cPickle.dump(hetrz_mea_sd_txt,txt_file)
    cPickle.dump(males_females_age_txt,txt_file)
    txt_file.close()

    pop_file = open('pop_'+str(n_stage)+'_run'+str(n_run)+'.txt','wb')
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
    cPickle.dump(gen_map,pop_file)
    pop_file.close()
