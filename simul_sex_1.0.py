# lightweight version - outputs age_distr every stage and rest only snapshot
# data, no sha_wie
## imports
from random import randint,uniform,choice,gauss,sample,shuffle
import time
import numpy as np
import cPickle

## prompts
#res_prompt = raw_input('\nResources as a function of population[pop] or a constant[const]? ')
#if res_prompt=='pop':
#    R = int(raw_input('Resource increment per stage: '))
#    k_var = float(raw_input('Resource regrowth factor: '))
#    res_upper_limit = int(raw_input('Resource upper limit: '))
#    start_res = int(raw_input('Starting amount of resources: '))
#else:
#    start_res = int(raw_input('Constant resources value: ')
#start_age_var = raw_input('All individuals start at reproduction age[y] or age is randomly distributed[random]? ')
#variance_var = float(raw_input('Variance index[0-1.4]: '))
#start_pop = int(raw_input('Starting number of individuals: '))
#number_of_stages = int(raw_input('Number of stages: '))
#crisis_stages = raw_input('Stages of crisis occurance[separate with a comma]: ')
#if crisis_stages!='':
#   crisis_stages = np.array(map(int,crisis_stages.split(',')))
#else:
#   crisis_stages = np.array([-1])
#sample_var = raw_input('Fraction of crisis survivors[0-1]: ')
#if sample_var!='':
#   sample_var = float(sample_var)
#max_death_rate = float(raw_input('Maximal death rate [default: 0.02]: '))
#surv_rate_distr = raw_input('Initial survival rate (distribution)[random or value in %]: ')
#repr_rate_distr = raw_input('Initial reproduction rate (distribution)[random or value in %]: ')
#max_repr_rate = float(raw_input('Maximal reproduction rate [default: 0.4]: '))
#recomb_rate_var = float(raw_input('Recombination rate [default: 0.01]: '))
#mut_rate_var = float(raw_input('Mutation rate [default: 0.001]: '))
#number_of_runs = int(raw_input('Number of runs: '))
#comment = raw_input('Comment: ')
#out = raw_input('\nWhat is the path to the output file?\n')

res_prompt = 'pop'
R = 1000
k_var = 1.6
res_upper_limit = 5000
start_res = 0
start_age_var = 'random'
variance_var = 0
start_pop = 500
number_of_stages = 200
crisis_stages = ''
if crisis_stages!='':
    crisis_stages = np.array(map(int,crisis_stages.split(',')))
else:
    crisis_stages = np.array([-1])
sample_var = '0.2'
if sample_var!='':
    sample_var = float(sample_var)
max_death_rate = 0.02
surv_rate_distr = 'random'
repr_rate_distr = 'random'
max_repr_rate = 0.4
recomb_rate_var = 0.01
mut_rate_var = 0.001
number_of_runs = 1
comment = 'test. 17-Jun-2015.'
out = '/home/arian/projects/model/output'

## constants
recombination_rate = recomb_rate_var
mutation_rate = mut_rate_var
number_of_bases = 1271
death_rate_increase = 3
number_positions = 20

## death rate, reproduction chance
death_rate_var = np.linspace(max_death_rate,0.001,21) # max to min
repr_rate_var = np.linspace(0,max_repr_rate,21) # min to max
## genome map
gen_map = range(0,71)+range(116,171)+[201]
shuffle(gen_map)
## plot variables
snapshot_plot_stages = np.around(np.linspace(0,number_of_stages,16),0)
ipx = np.arange(0,71)
surv_fit_var = np.linspace(0,1,21)
repr_fit_var = np.linspace(0,1,21)

## define functions
def chance(z):
    z = round(z * 1000000, 0)
    if (randint(1, 1000000) <= z):
        y = True
    else:
        y = False
    return y

def divers_function(x,n_base):
    if x>1.4:
        x = 1.4
    x_var = round(uniform(-x,x),2)
    if x_var>0:
        sigma_var = 1.4+round(-579.73676*x_var**7+2482.6373*x_var**6-4175.34457*x_var**5+3529.51159*x_var**4-1568.36701*x_var**3+344.84699*x_var**2-27.5476*x_var,2)
    else:
        sigma_var = 1.4+x_var
    a = np.array([gauss(mu=0.5, sigma=sigma_var) for i in range(n_base)], dtype=int)
    a = np.absolute(a)
    a[a>1] = 1
    a = list(a)
    return a

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def my_shuffle(array):
    shuffle(array)
    return array

# info.txt
time_print =  time.strftime('%X %x', time.gmtime())

info_file = open(out+'/info.txt', 'w')
if res_prompt=='pop':
    res_comm_var = '\n\nres_prompt = pop\nResource increment per stage: '+str(R)+'\nResource regrowth factor: '+str(k_var)+'\nResource upper limit: '+str(res_upper_limit)+'\nStarting amount of resources: '+str(start_res)
else:
    res_comm_var = '\n\nres_prompt = const\nResources: '+str(resources)
info_file.write('Version 1.0\nStarted at: '+time_print+'\n\nStarting age variable: '+start_age_var+'\nVariance index: '+str(variance_var)+'\nStarting number of individuals: '+str(start_pop)+'\nNumber of stages: '+str(number_of_stages)+'\nStages of crisis occurence: '+str(crisis_stages)+'\nFraction of crisis survivors: '+str(sample_var)+'\nMaximal death rate: '+str(max_death_rate)+'\nInitial survival rate (distribution)[%]: '+str(surv_rate_distr)+'\nMaximal reproduction rate: '+str(max_repr_rate)+'\nRecombination rate: '+str(recomb_rate_var)+'\nMutation rate var: '+str(mut_rate_var)+'\nNumber of runs: '+str(number_of_runs)+'\n\nComment: '+comment)
info_file.close()

# start
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
    if start_age_var=='y':
        if surv_rate_distr=='random' and repr_rate_distr=='random':
            start_age = 15
            for i in range(start_pop):
                individual = [start_age,[],[]]
                individual[1].append(choice((0, 1))) ## sex gene
                individual[2].append(0)
                individual[1] += divers_function(variance_var, number_of_bases-1)
                individual[2] += divers_function(variance_var, number_of_bases-1)
                population.append(individual)
        else:
            for i in range(start_pop):
                individual = [start_age,[],[]]
                individual[1].append(choice((0, 1))) ## sex gene
                individual[2].append(0)
                individual[1] += divers_function(variance_var, number_of_bases-1)
                individual[2] += divers_function(variance_var, number_of_bases-1)
                population.append(individual)

            if surv_rate_distr!='random':
                surv_rate_distr1 = (100-float(surv_rate_distr))/100
                surv_rate_distr2 = find_nearest(death_rate_var,surv_rate_distr1)
                srd3 = [1]*surv_rate_distr2+[0]*(20-surv_rate_distr2)
                for i in population:
                    k_count = 1
                    for k in gen_map:
                        my_shuffle(srd3)
                        if k<100:
                            for z in range(10):
                                i[1].pop(k_count)
                                i[2].pop(k_count)
                            for z in range(10):
                                i[1].insert(k_count, srd3[z])
                                i[2].insert(k_count, srd3[z+10])
                        k_count += 10

            if repr_rate_distr!='random':
                repr_rate_distr1 = find_nearest(repr_rate_var,float(repr_rate_distr)/100)
                srd3 = [1]*repr_rate_distr1+[0]*(20-repr_rate_distr1)
                for i in population:
                    k_count = 1
                    for k in gen_map:
                        my_shuffle(srd3)
                        if k>100 and k<200:
                            for z in range(10):
                                i[1].pop(k_count)
                                i[2].pop(k_count)
                            for z in range(10):
                                i[1].insert(k_count, srd3[z])
                                i[2].insert(k_count, srd3[z+10])
                        k_count += 10

    else:
        if surv_rate_distr=='random' and repr_rate_distr=='random':
            for i in range(start_pop):
                individual = [randint(0,70),[],[]]
                individual[1].append(choice((0, 1))) ## sex gene
                individual[2].append(0)
                individual[1] += divers_function(variance_var, number_of_bases-1)
                individual[2] += divers_function(variance_var, number_of_bases-1)
                population.append(individual)
        else:
            for i in range(start_pop):
                individual = [randint(0,70),[],[]]
                individual[1].append(choice((0, 1))) ## sex gene
                individual[2].append(0)
                individual[1] += divers_function(variance_var, number_of_bases-1)
                individual[2] += divers_function(variance_var, number_of_bases-1)
                population.append(individual)

            if surv_rate_distr!='random':
                surv_rate_distr1 = (100-float(surv_rate_distr))/100
                surv_rate_distr2 = find_nearest(death_rate_var,surv_rate_distr1)
                srd3 = [1]*surv_rate_distr2+[0]*(20-surv_rate_distr2)
                for i in population:
                    k_count = 1
                    for k in gen_map:
                        my_shuffle(srd3)
                        if k<100:
                            for z in range(10):
                                i[1].pop(k_count)
                                i[2].pop(k_count)
                            for z in range(10):
                                i[1].insert(k_count, srd3[z])
                                i[2].insert(k_count, srd3[z+10])
                        k_count += 10

            if repr_rate_distr!='random':
                repr_rate_distr1 = find_nearest(repr_rate_var,float(repr_rate_distr)/100)
                srd3 = [1]*repr_rate_distr1+[0]*(20-repr_rate_distr1)
                for i in population:
                    k_count = 1
                    for k in gen_map:
                        my_shuffle(srd3)
                        if k>100 and k<200:
                            for z in range(10):
                                i[1].pop(k_count)
                                i[2].pop(k_count)
                            for z in range(10):
                                i[1].insert(k_count, srd3[z])
                                i[2].insert(k_count, srd3[z+10])
                        k_count += 10

    ## starting population output
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

    ## stage loop
    for n_stage in range(0, number_of_stages+1):
        print n_stage
        
        ## extinction check
        if (len(population) == 0):
            pop_txt.append(len(population))
            res_txt.append(resources)
            print 'perished at stage '+str(n_stage)
            break

        ## PLOT VALUES OUTPUT
        # EVERY STAGE
        n_age = np.zeros((71,))

        ## output age_distr
        for i in population:
            for k in range(71):
                if i[0]==k:
                    n_age[k]+=1
                    break

        n_age = n_age/len(population)

        pop_txt.append(len(population))
        res_txt.append(resources)
        n_age_txt.append(n_age)

        ## ONLY 16 STAGES
        if n_stage in np.linspace(0,number_of_stages,16).astype(int):
            ## reseting output variables
            repr_rate_out = np.zeros((71,))
            repr_rate_junk_out = np.zeros((1,))
            death_rate_out = np.zeros((71,))
            death_rate_junk_out = np.zeros((1,))
            repr_rate_sd = [[]]*71
            surv_rate_sd = [[]]*71
            repr_fit = np.zeros((71,))
            repr_fit_junk = np.zeros((1,))
            surv_fit = np.zeros((71,))
            surv_fit_junk = np.zeros((1,))
            density_surv = np.zeros((21,))
            density_repr = np.zeros((21,))
            hetrz_mea = np.zeros((1260,)) # heterozigosity measure
            hetrz_mea_sd = [[]]*1260

            ## output genome
            for i in range(len(population)):
                down_limit = 1
                up_limit = 11
                for k in gen_map:
                    out_var = population[i][1][down_limit:up_limit].count(1)+population[i][2][down_limit:up_limit].count(1)
                    if k<100:
                        density_surv[out_var]+=1
                        death_rate_out[k]+=death_rate_var[out_var]
                        surv_rate_sd[k].append(1-death_rate_var[out_var]) # sd
                        surv_fit[k]+=surv_fit_var[out_var]
                        for t in range(10):
                            hetrz_mea[k*10+t] += population[i][1][down_limit+t]+population[i][2][down_limit+t]
                            hetrz_mea_sd[k*10+t].append(population[i][1][down_limit+t]+population[i][2][down_limit+t])
                    elif k<200:
                        density_repr[out_var]+=1
                        repr_rate_out[k-100]+=repr_rate_var[out_var]
                        repr_rate_sd[k-100].append(repr_rate_var[out_var]) # sd
                        repr_fit[k-100]+=repr_fit_var[out_var]
                        for t in range(10):
                            hetrz_mea[(k-116)*10+710+t] += population[i][1][down_limit+t]+population[i][2][down_limit+t]
                            hetrz_mea_sd[(k-116)*10+710+t].append(population[i][1][down_limit+t]+population[i][2][down_limit+t])
                    else:
                        death_rate_junk_out[0]+=death_rate_var[out_var]
                        surv_fit_junk[0]+=surv_fit_var[out_var]
                        repr_rate_junk_out[0]+=repr_rate_var[out_var]
                        repr_fit_junk[0]+=repr_fit_var[out_var]
                    down_limit = up_limit
                    up_limit += 10

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

        # everyone gets 1 year older
        for i in range(len(population)):
            population[i][0] += 1

        ### RESOURCES
        if res_prompt=='pop': # function of population
            if (len(population) > resources):
                k = 1
            else:
                k = k_var
            resources = int((resources-len(population))*k+R) ## Res(s) = (Res(s-1)-N)*k+R
            ## resources cannot be negative or higher than x
            if (resources < 0):
                resources = 0
            elif (resources > res_upper_limit):
                resources = res_upper_limit
            ## x increases the death rate when resources are zero
            if (resources == 0):
                x = x*death_rate_increase
            else:
                x = 1.0

        else: # constant
            if len(population)>resources:
                x = x*death_rate_increase
            else:
                x = 1.0

        ## adult sorting and genome data transcription so that parent genome remains unchanged after reproduction 
        adult = []
        for i in range(len(population)):
            if (population[i][0] > 15):
                u = []
                u.append(population[i][0])
                u.append(list(population[i][1]))
                u.append(list(population[i][2]))
                adult.append(u)

        ## adult selection
        adult_pass = []
        for i in range(len(adult)):
            repr_rate = 0 # in case age>70
            down_limit = 1
            up_limit = 11
            for k in gen_map:
                if k>100 and k<200:
                    if adult[i][0]==k-100:
                        repr_rate = repr_rate_var[adult[i][1][down_limit:up_limit].count(1)+adult[i][2][down_limit:up_limit].count(1)]
                        break
                down_limit = up_limit
                up_limit += 10
            if chance(repr_rate):
                adult_pass.append(adult[i])

        ## separating males/females
        adult_males=[]
        adult_females=[]
        for i in adult_pass:
            if i[1][0] == 1:
                adult_males.append(i)
            else:
                adult_females.append(i)
                
        ## making pairs
        pairs_exc = abs(len(adult_males)-len(adult_females))
        # the excess of males/females is eliminated
        if len(adult_males)>len(adult_females):
            shuffle(adult_males)
            for i in range(pairs_exc):
                adult_males.pop()
        else:
            shuffle(adult_females)
            for i in range(pairs_exc):
                adult_females.pop()
        pairs = zip(adult_males,adult_females)

        ## creating a new individual
        if len(adult_males)!=0 and len(adult_females)!=0:
            for n in pairs: # every selected individual reproduces
                parent_1 = n[0]
                parent_2 = n[1]

                ## recombination with recombination_rate chance
                for z in range(2, number_of_bases+1):
                    if chance(recombination_rate): 
                        w = parent_1[1][1:z]
                        r = parent_1[2][1:z]
                        parent_1[1] = [parent_1[1][0]]+r+parent_1[1][z:]
                        parent_1[2] = [parent_1[2][0]]+w+parent_1[2][z:]

                for z in range(2, number_of_bases+1):
                    if chance(recombination_rate): 
                        w = parent_2[1][1:z]
                        r = parent_2[2][1:z]
                        parent_2[1] = [parent_2[1][0]]+r+parent_2[1][z:]
                        parent_2[2] = [parent_2[2][0]]+w+parent_2[2][z:]

                choice_chromatide_parent_1 = choice(parent_1[1:3])
                choice_chromatide_parent_2 = choice(parent_2[1:3])

                ## mutation
                for i in range(1, number_of_bases):
                    if chance(mutation_rate): 
                        if (choice_chromatide_parent_1[i] == 0):
                            if chance(0.1):
                                choice_chromatide_parent_1[i] = 1
                        else:
                            choice_chromatide_parent_1[i] = 0
                       
                    if chance(mutation_rate):
                        if (choice_chromatide_parent_2[i] == 0):
                            if chance(0.1):
                                choice_chromatide_parent_2[i] = 1
                        else:
                            choice_chromatide_parent_2[i] = 0

                new_born = [0]
                new_born.append(choice_chromatide_parent_1)
                new_born.append(choice_chromatide_parent_2)
                population.append(new_born)

        ## dying    
        j = 0
        for i in range(len(population)):
            death_rate = 1 # in case age>70
            down_limit = 1
            up_limit = 11
            for k in gen_map:
                if k<100:
                    if population[i-j][0]==k:
                        death_rate = death_rate_var[population[i-j][1][down_limit:up_limit].count(1)+population[i-j][2][down_limit:up_limit].count(1)]
                        break
                down_limit = up_limit
                up_limit += 10
                        
            if chance(death_rate*x):
                population.pop(i-j)
                j += 1

        ## enviromental crisis
        if any(crisis_stages==n_stage):
            population = sample(population, int(sample_var*len(population)))

    ## RUN ENDED

    ## pop-pyramid output
    males_age = np.zeros((71,))
    females_age = np.zeros((71,))

    for i in population:
        if i[1][0]==1:
            for k in range(71):
                if i[0]==k:
                    males_age[k]+=1
                    break
        else:
            for k in range(71):
                if i[0]==k:
                    females_age[k]+=1
                    break

    males_age = males_age/len(population)*100
    females_age = females_age/len(population)*100
    males_females_age_txt = [list(males_age), list(females_age)]

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

## ended
