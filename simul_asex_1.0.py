# lightweight version - outputs age_distr every stage and rest only snapshot data, no sha_wie

print "Initialising...",

## IMPORT LIBRARIES ##
from random import randint,uniform,choice,gauss,sample,shuffle
import time
import numpy as np
import cPickle
import copy
import itertools
import simul_functions as fn

## PARAMETERS ## 
res_prompt = 'pop' # Resource growth: population-dependent vs constant
R = 1000 # Resource increment per stage
k_var = 1.6 # Resource regrowth factor
res_upper_limit = 5000 # Max resources
start_res = 0 # Starting resources
start_age_var = 'random' # All individuals start at reproduction age ("y", or distribution is random ("random"):
variance_var = 1.4 # "Variance index"?
start_pop = 5000 # Starting population size
number_of_stages = 200 # Total number of stages
crisis_stages = '' # Stages of extrinsic death crisis
if crisis_stages!='':
    crisis_stages = np.array(map(int,crisis_stages.split(',')))
else:
    crisis_stages = np.array([-1])
sample_var = '' # Fraction of crisis survivors (if applicable)
if sample_var!='':
    sample_var = float(sample_var)
max_death_rate = 0.02 # Max death rate?
surv_rate_distr = 'random' # Initial survival rate distribution (random or % value)
repr_rate_distr = 'random' # Initial reproduction rate distribution (random or % value)
max_repr_rate = 0.2 # half the value of sex_model because here every individual has one offspring
recomb_rate_var = 0.01 # Recombination rate
mut_rate_var = 0.001 # Mutation rate
number_of_runs = 1
comment = 'test'
out = "testrun/" 

## constants
recombination_rate = recomb_rate_var
mutation_rate = mut_rate_var
number_of_bases = 1270 # no sex gene
death_rate_increase = 3
number_positions = 20

## death rate, reproduction chance
death_rate_var = np.linspace(max_death_rate,0.001,21) # max to min
repr_rate_var = np.linspace(0,max_repr_rate,21) # min to max
## genome map: survival (0 to 70), reproduction (16 to 70), neutral (single)
gen_map = np.asarray(range(0,71)+range(116,171)+[201])
## plot variables
snapshot_plot_stages = np.around(np.linspace(0,number_of_stages,16),0)
ipx = np.arange(0,71)
surv_fit_var = np.linspace(0,1,21)
repr_fit_var = np.linspace(0,1,21)

# Write info file.
time_print =  time.strftime('%X %x', time.gmtime())

info_file = open(out+'info.txt', 'w')
if res_prompt=='pop':
    res_comm_var = '\n\nres_prompt = pop\nResource increment per stage: '+str(R)+'\nResource regrowth factor: '+str(k_var)+'\nResource upper limit: '+str(res_upper_limit)+'\nStarting amount of resources: '+str(start_res)
else:
    res_comm_var = '\n\nres_prompt = const\nResources: '+str(resources)
info_file.write('Version 1.0\nStarted at: '+time_print+'\n\nStarting age variable: '+start_age_var+'\nVariance index: '+str(variance_var)+'\nStarting number of individuals: '+str(start_pop)+'\nNumber of stages: '+str(number_of_stages)+'\nStages of crisis occurence: '+str(crisis_stages)+'\nFraction of crisis survivors: '+str(sample_var)+'\nMaximal death rate: '+str(max_death_rate)+'\nInitial survival rate (distribution)[%]: '+str(surv_rate_distr)+'\nMaximal reproduction rate: '+str(max_repr_rate)+'\nRecombination rate: '+str(recomb_rate_var)+'\nMutation rate var: '+str(mut_rate_var)+'\nNumber of runs: '+str(number_of_runs)+'\n\nComment: '+comment)
info_file.close()

################
## SIMULATION ##
################
print "done."
for n_run in range(1, number_of_runs+1):
    print ""
    print "Beginning run "+str(n_run)+"."
    print ""
    np.random.shuffle(gen_map) # reshuffle genome every run
    ## constants and lists
    resources = start_res
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
    for i in range(start_pop-1):
        indiv = fn.make_individual(start_age_var, variance_var,
                number_of_bases, gen_map, surv_rate_distr, 
                repr_rate_distr)
        population.append(indiv)
    population = np.array(population)
    print "done."

    ## starting population output:
    pop_file = open(out+'pop_0_run'+str(n_run)+'.txt','wb')
    cPickle.dump(population,pop_file)
    cPickle.dump(resources,pop_file)
    if res_prompt=='pop':
        cPickle.dump(R,pop_file)
        cPickle.dump(k_var,pop_file)
        cPickle.dump(res_upper_limit,pop_file)
    cPickle.dump(max_death_rate,pop_file)
    cPickle.dump(max_repr_rate,pop_file)
    cPickle.dump(recomb_rate_var,pop_file)
    cPickle.dump(mut_rate_var,pop_file)
    cPickle.dump(gen_map,pop_file)
    pop_file.close()

    # # # # # # # # #
    # 1: STAGE LOOP #
    # # # # # # # # # 
    print "Beginning stage loop."
    for n_stage in range(0, number_of_stages+1):

        N = len(population)
        if(N==0):
            print "Perished at stage "+str(n_stage)+"."
            break
        else:
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
        if res_prompt=='pop': # function of population
            resources = fn.update_resources(resources, N, R, k_var, 
                    res_upper_limit, True)
            x = x*death_rate_increase if resources == 0 else 1.0
        else: # constant; death rate increases if population exceeds
            x = x*death_rate_increase if N>resources else 1.0

        # Reproduction
        population = fn.reproduction_asex(population, N, gen_map,
                number_of_bases, repr_rate_var, mutation_rate, True)
        N = len(population)

        # Death
        population = fn.death(population, N, gen_map, number_of_bases,
                death_rate_var, x, True)

        # Extrinsic death crisis:
        if n_stage in crisis_stages:
            N = len(population)
            n_survivors = int(N*sample_var)
            population = population[sample(range(len(N)), n_survivors)]
            print "Extrinsic death crisis! "+str(len(population))+"individuals survived."

    ## RUN ENDED

    ## pop-pyramid output
    males_females_age = n_age*100
    males_females_age_txt = list(males_females_age)

    ## text file
    txt_file =  open(out+'/plot_values_run'+str(n_run)+'.txt','wb')
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

    pop_file = open(out+'pop_'+str(n_stage)+'_run'+str(n_run)+'.txt','wb')
    cPickle.dump(population,pop_file)
    cPickle.dump(resources,pop_file)
    if res_prompt=='pop':
        cPickle.dump(R,pop_file)
        cPickle.dump(k_var,pop_file)
        cPickle.dump(res_upper_limit,pop_file)
    cPickle.dump(max_death_rate,pop_file)
    cPickle.dump(max_repr_rate,pop_file)
    cPickle.dump(recomb_rate_var,pop_file)
    cPickle.dump(mut_rate_var,pop_file)
    cPickle.dump(gen_map,pop_file)
    pop_file.close()
