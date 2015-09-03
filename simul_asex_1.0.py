# lightweight version - outputs age_distr every stage and rest only snapshot data, no sha_wie

## IMPORT LIBRARIES ##
from random import randint,uniform,choice,gauss,sample,shuffle
import time
import numpy as np
import cPickle
import copy
import itertools
import simul_functions

## PARAMETERS ## 
res_prompt = 'pop' # Resource growth: population-dependent vs constant
R = 1000 # Resource increment per stage
k_var = 1.6 # Resource regrowth factor
res_upper_limit = 5000 # Max resources
start_res = 0 # Starting resources
start_age_var = 'random' # All individuals start at reproduction age ("y", or distribution is random ("random"):
variance_var = 1.4 # "Variance index"?
start_pop = 500 # Starting population size
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
out = '/home/arian/projects/model/output' # Output path

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
np.random.shuffle(gen_map)
## plot variables
snapshot_plot_stages = np.around(np.linspace(0,number_of_stages,16),0)
ipx = np.arange(0,71)
surv_fit_var = np.linspace(0,1,21)
repr_fit_var = np.linspace(0,1,21)

# Write info file.
time_print =  time.strftime('%X %x', time.gmtime())

info_file = open(out+'/info.txt', 'w')
if res_prompt=='pop':
    res_comm_var = '\n\nres_prompt = pop\nResource increment per stage: '+str(R)+'\nResource regrowth factor: '+str(k_var)+'\nResource upper limit: '+str(res_upper_limit)+'\nStarting amount of resources: '+str(start_res)
else:
    res_comm_var = '\n\nres_prompt = const\nResources: '+str(resources)
info_file.write('Version 1.0\nStarted at: '+time_print+'\n\nStarting age variable: '+start_age_var+'\nVariance index: '+str(variance_var)+'\nStarting number of individuals: '+str(start_pop)+'\nNumber of stages: '+str(number_of_stages)+'\nStages of crisis occurence: '+str(crisis_stages)+'\nFraction of crisis survivors: '+str(sample_var)+'\nMaximal death rate: '+str(max_death_rate)+'\nInitial survival rate (distribution)[%]: '+str(surv_rate_distr)+'\nMaximal reproduction rate: '+str(max_repr_rate)+'\nRecombination rate: '+str(recomb_rate_var)+'\nMutation rate var: '+str(mut_rate_var)+'\nNumber of runs: '+str(number_of_runs)+'\n\nComment: '+comment)
info_file.close()

################
## SIMULATION ##
################

for n_run in range(1, number_of_runs+1):

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
    for i in range(start_pop):
        start_age = 15 if start_age_var=="y" else randint(0,70) # Uniform?
        individual = [start_age,[],[]]
        individual[1] += starting_genome(variance_var, number_of_bases, 
                gen_map, surv_rate_distr, repr_rate_distr)
        individual[2] += starting_genome(variance_var, number_of_bases,
                gen_map, surv_rate_distr, repr_rate_distr)
        population.append(individual)

    ## starting population output::
    pop_file = open(out+'/pop_0_run'+str(n_run)+'.txt','wb')
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

    for n_stage in range(0, number_of_stages+1):
        
        pop_txt.append(len(population))
        res_txt.append(resources)

        if(len(population)==0):
            print "perished at stage "+str(n_stage)
            break
        else:
            print n_stage

        # Get (proportional) age distribution:
        n_age = np.bincount([i[0] for i in population])/len(population)

        n_age_txt.append(n_age)

        ## ONLY 16 STAGES
        if n_stage in np.linspace(0,number_of_stages,16).astype(int):
            ## reseting output variables
            repr_rate_out = np.zeros((71,))
            repr_rate_junk_out = np.zeros((1,))
            death_rate_out = np.zeros((71,))
            death_rate_junk_out = np.zeros((1,))
            # sd-collect genome values for every age
            repr_rate_sd = [[]]*71
            surv_rate_sd = [[]]*71
            repr_fit = np.zeros((71,))
            repr_fit_junk = np.zeros((1,))
            surv_fit = np.zeros((71,))
            surv_fit_junk = np.zeros((1,))
            density_surv = np.zeros((21,))
            density_repr = np.zeros((21,))
            hetrz_mea = np.zeros((1260,)) # heterozigosity measure
            hetrz_mea_sd = [[]]*1260 # heterozigosity measure sd
            
            neut_locus = np.nonzero(gen_map==201)[0][0] # Position of neutral locus on gen_map
            neut_pos = (neut_locus*10, (neut_locus+1)*10) # Positions in genome array corresponding to neutral locus
            for i in range(len(population)):
                I = population[i] 
                surv_locus = np.nonzero(gen_map==I[0])[0][0] # Position of age-appropriate survival locus on gen_map
                repr_locus = np.nonzero(gen_map==(I[0]+100))[0][0] # Position of age-appropriate reproduction locus on gen_map
                surv_pos = (surv_locus*10, (surv_locus+1)*10)  # Positions in genome array corresponding to survival locus
                repr_pos = (repr_locus*10, (repr_locus+1)*10)  # Positions in genome array corresponding to reproduction locus
                # Get genotype sums:
                surv_out = I[1][surv_pos[0]:surv_pos[1]].count(1) + I[2][surv_pos[0]:surv_pos[1]].count(1)  
                repr_out = I[1][repr_pos[0]:repr_pos[1]].count(1) + I[2][repr_pos[0]:repr_pos[1]].count(1)  
                neut_out = I[1][neut_pos[0]:neut_pos[1]].count(1) + I[2][neut_pos[0]:neut_pos[1]].count(1)  
                # Survival statistics:
                density_surv[surv_out] += 1
                death_rate_out[surv_locus] += death_rate_var[surv_out]
                surv_rate_sd[surv_locus].append(1-death_rate_var[surv_out])
                surv_fit[surv_locus] += surv_fit_var[surv_out]
                for t in range(surv_pos[0], surv_pos[1]):
                    hetrz = I[1][t]+I[2][t]
                    hetrz_mea[t] += hetrz
                    hetrz_mea_sd[t].append(hetrz)
                # Reproduction statistics:
                density_repr[repr_out] += 1
                repr_rate_out[repr_locus] += repr_rate_var[repr_out]
                repr_rate_sd[repr_locus].append(repr_rate_var[repr_out])
                repr_fit[repr_locus] += repr_fit_var[repr_out]
                for t in range(repr_pos[0], repr_pos[1]):
                    hetrz = I[1][t]+I[2][t]
                    hetrz_mea[t] += hetrz
                    hetrz_mea_sd[t].append(hetrz)
                # Neutral statistics
                death_rate_junk_out[0] += death_rate_var[neut_out]
                surv_fit_junk[0] += surv_fit_var[neut_out]
                repr_rate_junk_out[0]+=repr_rate_var[neut_out]
                repr_fit_junk[0]+=repr_fit_var[neut_out]

            ## average the output data 
            surv_rate_out = 1-death_rate_out/len(population)
            surv_rate_junk_out = 1-death_rate_junk_out/len(population)
            repr_rate_out = repr_rate_out/len(population)
            repr_rate_junk_out = repr_rate_junk_out/len(population)
            surv_fit = surv_fit/len(population)
            surv_fit_junk = surv_fit_junk/len(population)
            repr_fit = repr_fit/len(population)
            repr_fit_junk = repr_fit_junk/len(population)
            density = (density_surv+density_repr)/(126*len(population))
            density_surv = density_surv/(71*len(population))
            density_repr = density_repr/(55*len(population))
            hetrz_mea = hetrz_mea/(2*len(population))

            ## standard deviation
            for i in range(71):
                surv_rate_sd[i] = np.sqrt(np.mean(np.square(surv_rate_sd[i]-surv_rate_out[i])))
                repr_rate_sd[i] = np.sqrt(np.mean(np.square(repr_rate_sd[i]-repr_rate_out[i])))
            
            for i in range(1260):
                hetrz_mea_sd[i] = np.sqrt(np.mean(np.square(hetrz_mea_sd[i]-hetrz_mea[i])))

            ## append to text file
            repr_rate_txt.append(repr_rate_out)
            repr_rate_sd_txt.append(repr_rate_sd) # sd
            repr_rate_junk_txt.append(repr_rate_junk_out)
            surv_rate_txt.append(surv_rate_out)
            surv_rate_sd_txt.append(surv_rate_sd) # sd
            surv_rate_junk_txt.append(surv_rate_junk_out)
            repr_fit_txt.append(repr_fit)
            repr_fit_junk_txt.append(repr_fit_junk)
            surv_fit_txt.append(surv_fit)
            surv_fit_junk_txt.append(surv_fit_junk)
            density_surv_txt.append(density_surv)
            density_repr_txt.append(density_repr)
            hetrz_mea_txt.append(hetrz_mea)
            hetrz_mea_sd_txt.append(hetrz_mea_sd)

        # everyone gets 1 stage older
        for i in range(len(population)):
            population[i][0] += 1

        # Change in resources
        # If res_prompt=='pop', resources are a function of population size and regrow:
        # Each individuals consumes 1 resource per stage; if resources are left
        # Multiply them by regrowth factor and add to set increment factor; else
        # subtract deficit from increment factor. Death rate increases if resources
        # run out.
        #
        # Else, resources are constant and death rate increases if there are
        # more individuals than available resources.

        if res_prompt=='pop': # function of population
            k = 1 if (len(population) > resources) else k_var
            resources = int((resources-len(population))*k+R) # Res_{t+1} = (Res_t-N)*k+R
            resources = min(max(resources, 0), res_upper_limit) # resources cannot be negative or exceed limit
            # If resources are 0, death rate increases
            x = x*death_rate_increase if resources == 0 else 1.0
        else: # constant
            x = x*death_rate_increase if len(population)>resources else 1.0
        ### So death rate increase compounds over multiple stages?

        ## adult sorting and genome data transcription so that parent genome remains unchanged after reproduction 
        which_adults = np.array([item[0] for item in pop])>15
        adult = copy.deepcopy(list(itertools.compress(population, which_adults)))

        ## parent selection
        adult_pass = []
        for i in range(len(adult)):
            I = adult[i]
            if I[0] <= 70:
                locus = np.nonzero((gen_map==I[0]+100)*1)[0][0] # Get position on gen_map corresponding to reproductive locus for this age
                pos = (locus*10, (locus+1)*10) # Positions in the genome array to use for calculating reproductive probability.
                repr_rate = repr_rate_var[ I[1][pos[0]:pos[1]].count(1) + I[2][pos[0]:pos[1]].count(1) ]
                # Reproduction rate = min_rate + (max_rate-min_rate)/21 * locus_sum (across both chromosomes)
                if chance(repr_rate):
                    adult_pass.append(I)
        
        # Make new individuals from parents
        for a in adult_pass: # every selected individual reproduces
            chr1 = a[1]
            chr2 = a[2]
            ## mutation
            for i in range(0, number_of_bases):
                mu = np.array([[chance(mutation_rate), 1-chance(0.1*mutation_rate)], 
                    [chance(mutation_rate), 1-chance(0.1*mutation_rate)]])
                # 0 -> 1 mutationts 10x as likely as 1 -> 0
                chr1[i] = mu[0, chr1[i]]
                chr2[i] = mu[1, chr1[i]]
            population.append([0, itertools.copy(chr1), itertools.copy(chr2)]) # Add newborn to population

        ## death selection
        j = 0
        for i in range(len(population)):
            I = population[i]
            if I[0] <= 70:
                locus = np.nonzero((gen_map==I[0])*1)[0][0] # Get position on gen_map corresponding to survival locus for this age
                pos = (locus*10, (locus+1)*10) # Positions in the genome array to use for calculating survival probability.
                death_rate = death_rate_var[ I[1][pos[0]:pos[1]].count(1) + I[2][pos[0]:pos[1]].count(1) ]
                # Death rate = max_rate - (max_rate-min_rate)/21 * locus_sum (across both chromosomes)
            else: death_rate = 1
            if chance(death_rate*x): # Genetic death rate compounded by resource shortage
                population.pop(i-j)
                j += 1

        ## extrinsic death crisis
        if any(crisis_stages==n_stage):
            population = sample(population, int(sample_var*len(population)))

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

    pop_file = open(out+'/pop_'+str(n_stage)+'_run'+str(n_run)+'.txt','wb')
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
