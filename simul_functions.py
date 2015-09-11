import scipy.stats
import numpy as np
from random import randint
from copy import copy,deepcopy
import time
import os
from importlib import import_module
import sys
import cPickle

rand = scipy.stats.uniform(0,1) # Generate random number generator

def chance(z,n=1):
    return rand.rvs(n)<z

def starting_genome(var,n,gen_map,s_dist,r_dist):
    ### Returns a binary array of length n, with the proportion of 1's
    ### determined by the initial distribution specified (random or
    ### a constant percentage).
    if n % len(gen_map) != 0:
        raise ValueError("Genome length must be integer multiple of map length.")
    q = n/len(gen_map)
    var=min(1.4, var)
    sd = var**0.5
    p=scipy.stats.truncnorm(-0.5/sd, 0.5/sd, loc=0.5, scale=sd).rvs(1) 
    # 0/1-truncated normal distribution with mean 0.5 and sd as given
    a = (rand.rvs(n)<p)*1
    # If survival and/or reproduction probability is not random, 
    # replace appropriate loci in genome with new 1/0 distribution.
    if s_dist != "random":
        s = float(s_dist)/100
        s_loci = np.nonzero(gen_map<100)[0]
        s_pos = []
        for x in s_loci:
            s_pos.append(range(x*q, (x+1)*q))
        a[s_pos] = chance(s, len(s_pos))
    if r_dist != "random":
        r = float(r_dist)/100
        r_loci = np.nonzero(np.logical_and(gen_map>100,gen_map<200))[0]
        a[r_loci] = rand.rvs(len(r_loci))<r
        r_pos = []
        for x in r_loci:
            r_pos.append(range(x*q, (x+1)*q))
        a[r_pos] = chance(s, len(r_pos))
    return list(a)

def make_individual(age_random, var, n, gen_map, s_dist, r_dist):
    ### Returns a [1,2n+1]-dimensional array representing an individual
    ### with a starting age and two length-n chromosomes
    age = randint(0,70) if age_random else 15
    chr1 = starting_genome(var, n, gen_map, s_dist, r_dist)
    chr2 = starting_genome(var, n, gen_map, s_dist, r_dist)
    individual = np.concatenate(([age], chr1, chr2))
    return individual

def make_population(start_pop, age_random, variance, chr_len, gen_map,
        s_dist, r_dist):
    print "Generating starting population...",
    population = []
    for i in range(start_pop-1):
        indiv = make_individual(age_random, variance, chr_len, 
                gen_map, s_dist, r_dist)
        population.append(indiv)
    population = np.array(population)
    print "done."
    return population

def update_resources(res0, N, R, V, limit, verbose=False):
    if verbose: print "Updating resources...",
    k = 1 if N>res0 else V
    res1 = int((res0-N)*k+R)
    res1 = min(max(res1, 0), limit)
    # Resources can't be negative or exceed limit.
    if verbose: print "Done. "+str(res0)+" -> "+str(res1)
    return res1

def reproduction_asex(population, maturity, max_ls, gen_map, chr_len, 
        r_range, m_rate, verbose=False):
    if verbose: print "Calculating reproduction...",
    parents = np.empty(0, int)
    ages = population[:,0]
    for age in range(maturity, min(max(ages), max_ls-1)):
        # Get indices of reproductive locus
        locus = np.nonzero(gen_map==(age+100))[0][0]
        pos = np.arange(locus*10, (locus+1)*10)+1
        # Get sub-population of correct age and subset genome to locus
        match = (ages == age)
        which = np.nonzero(match)[0]
        pop = population[match][:,np.append(pos, pos+chr_len)]
        gen = np.sum(pop, axis=1)
        # Sum over reproductive locus for every individual of that age
        # to get reprodctive genotype for each
        repr_rates = r_range[gen]
        # Convert reproductive genotypes into probabilities
        # Determine parents in subpopulation and add indices to record
        newparents = which[chance(repr_rates, len(repr_rates))]
        parents = np.append(parents, newparents)
    # Create and mutate children
    children = deepcopy(population[parents])
    children[children==0]=chance(m_rate,np.sum(children==0))
    children[children==1]=1-chance(0.1*m_rate, np.sum(children==1))
    children[:,0] = 0 # Make newborn
    population=np.vstack([population,children]) # Add to population
    if verbose: print "done. "+str(len(children))+" new individuals born."
    return(population)

def death(population, max_ls, gen_map, chr_len, 
        d_range, x, verbose=False):
    if verbose: print "Calculating death...",
    survivors = np.empty(0, int)
    ages = population[:,0]
    for age in range(min(max(ages), max_ls-1)):
        # Get indices of survival locus
        locus = np.nonzero(gen_map==age)[0][0]
        pos = np.arange(locus*10, (locus+1)*10)+1
        # Get sub-population of correct age and subset genome to locus
        match = (ages == age)
        which = np.nonzero(match)[0]
        pop = population[match][:,np.append(pos, pos+chr_len)]
        gen = np.sum(pop, axis=1)
        # Sum over survival locus for every individual of that age
        # to get survival genotype for each
        surv_rates = (1-d_range[gen]*x)
        # Convert survival genotypes into probabilities (penalised by
        # starvation factor)
        # Determine survivors in subpopulation and add indices to record
        newsurvivors = which[chance(surv_rates, len(surv_rates))]
        survivors = np.append(survivors, newsurvivors)
    new_population = population[survivors]
    if verbose:
        dead = len(population) - len(survivors)
        print "done. "+str(dead)+" individuals died."
    return(new_population)

def yesno(q):
    yes = ["y", "Y", "yes", "Yes", "YES"]
    no = ["n", "N", "no", "No", "NO"]
    while True:
        var = raw_input(q+" ").strip()
        if q in yes: 
            return True
        elif q in no: 
            return False
        else:
            print "Invalid input.\n"
def getnumber(q, cl, default=""):
    while True:
        var = raw_input(q+" ").strip()
        if var == "":
            var = default
        try:
            return(cl(var))
        except ValueError:
            print "Invalid input.\n"

def getconf(argv):
    loop = True
    while loop:
        if len(argv)>1:
            conf = argv[2]
        else:
            conf = raw_input("Name of config file in working directory:  ")
        try:
            p = import_module(conf)
            loop = False
        except ImportError:
            if len(argv)>1:
                sys.exit("ImportError: No such file in working directory.")
            else:
                print "No such file in working directory."
    print "Config file: "+conf+".py"
    return p

def getwd(argv):
    loop = True
    while loop:
        if len(argv)>1:
            path = argv[1]
        else:
            path = raw_input("Path to working directory:  ")
        try:
            sys.path.remove(os.getcwd())
            os.chdir(path)
            loop = False
        except ImportError:
            if len(argv)>1:
                sys.exit("OSError: Invalid path to working directory.")
            else:
                print "Invalid path to working directory."
    sys.path.append(os.getcwd())
    print "Working directory: "+os.getcwd()

def initialise_record(m, n_stages, max_ls, n_bases, chr_len, window_size):
    """ Create a new record dictionary object for recording output 
    data from simulation """
    array1 = np.zeros([m,max_ls])
    array2 = np.zeros([m,2*n_bases+1])
    array3 = np.zeros(m)
    array4 = np.zeros(n_stages)
    record = {
        "population_size":copy(array4),
        "resources":copy(array4),
        "starvation_factor":copy(array4),
        "age_distribution":copy(array1),
    	"surv_mean":copy(array1),
    	"surv_sd":copy(array1),
    	"repr_mean":copy(array1),
    	"repr_sd":copy(array1),
    	"density_surv":copy(array2),
    	"density_repr":copy(array2),
    	"n1":np.zeros([m,chr_len]),
    	"s1":np.zeros([m,chr_len-window_size+1]),
    	"fitness":copy(array1),
    	"entropy":copy(array3),
    	"surv_junk":copy(array3),
    	"repr_junk":copy(array3),
    	"fitness_junk":copy(array3)
        }
    return record

def quick_update(record, n_stage, N, resources, x):
    record["population_size"][n_stage] = N
    record["resources"][n_stage] = resources
    record["starvation_factor"][n_stage] = x
    return(record)

def x_bincount(array):
    return np.bincount(array, minlength=3)

def moving_average(a, n):
    c = np.cumsum(a, dtype=float)
    c[n:] = c[n:]-c[:-n]
    return c[(n-1):]/n

def update_record(record, population, N, resources, x, gen_map, chr_len, 
        n_bases, d_range, r_range, maturity, max_ls, window_size, 
        n_stage, n_snap):

    record = quick_update(record, n_stage, N, resources, x)
    ages = population[:,0]
    b = n_bases # Binary units per locus

    ## AGE-DEPENDENT STATS ##
    # Genotype sum distributions:
    density_surv = np.zeros((2*b+1,))
    density_repr = np.zeros((2*b+1,))
    # Mean death/repr rates by age:
    surv_mean = np.zeros(max_ls)
    repr_mean = np.zeros(max_ls)
    # Death/repr rate SD by age:
    surv_sd = np.zeros(max_ls)
    repr_sd = np.zeros(max_ls)
    # Loop over ages:
    for age in range(max(ages)):
        pop = population[ages==age]
        # Find loci and binary units:
        surv_locus = np.nonzero(gen_map==age)[0][0]
        surv_pos = np.arange(surv_locus*b, (surv_locus+1)*b)+1
        # Subset array to relevant columns and find genotypes:
        surv_pop = pop[:,np.append(surv_pos, surv_pos+chr_len)]
        surv_gen = np.sum(surv_pop, axis=1)
        # Find death/reproduction rates:
        surv_rates = 1-d_range[surv_gen]
        # Calculate statistics:
        surv_mean[age] = np.mean(surv_rates)
        surv_sd[age] = np.std(surv_rates)
        density_surv += np.bincount(surv_gen, minlength=2*b+1)
        if age>=maturity:
            # Same for reproduction if they're adults
            repr_locus = np.nonzero(gen_map==(age+100))[0][0]
            repr_pos = np.arange(repr_locus*b, (repr_locus+1)*b)+1
            repr_pop = pop[:,np.append(repr_pos, repr_pos+chr_len)]
            repr_gen = np.sum(repr_pop, axis=1)
            repr_rates = r_range[repr_gen]
            repr_mean[age] = np.mean(repr_rates)
            repr_sd[age] = np.std(repr_rates)
            density_repr += np.bincount(repr_gen, minlength=2*b+1)
    # Average densities over whole population
    density_surv /= float(N)
    density_repr /= float(N)
    # Calculate per-age average genomic fitness
    x_surv = np.cumprod(surv_mean)
    fitness = np.cumsum(x_surv * repr_mean)

    ## AGE-INVARIANT STATS ##
    # Frequency of 1's at each position on chromosome:
    # Average over array, then average at each position over chromosomes
    n1s = np.sum(population, 0)/float(N)
    n1 = (n1s[np.arange(chr_len)+1]+n1s[np.arange(chr_len)+chr_len+1])/2 
    # Standard deviation of 1 frequency over sliding window
    w = window_size
    s1 = np.sqrt(moving_average(n1**2, w)-moving_average(n1, w)**2)
    # Shannon-Weaver entropy over entire genome population
    gen = population[1:]
    p1 = np.sum(gen)/np.size(gen)
    entropy = scipy.stats.entropy(np.array([1-p1, p1]))
    # Junk stats calculated from neutral locus
    neut_locus = np.nonzero(gen_map==201)[0][0] 
    neut_pos = np.arange(neut_locus*b, (neut_locus+1)*b)+1
    neut_pop = population[:,np.append(neut_pos, neut_pos+chr_len)]
    neut_gen = np.sum(neut_pop, axis=1)
    surv_junk = np.mean(1-d_range[neut_gen])
    repr_junk = np.mean(r_range[neut_gen]) # Junk SDs?
    fitness_junk = np.cumsum(np.cumprod(surv_junk) * repr_junk)

    ## APPEND RECORD OBJECT ##
    record["age_distribution"][n_snap] = np.bincount(ages, 
            minlength = max_ls)/float(N)
    record["surv_mean"][n_snap] = surv_mean
    record["surv_sd"][n_snap] = surv_sd
    record["repr_mean"][n_snap] = repr_mean
    record["repr_sd"][n_snap] = repr_sd
    record["density_surv"][n_snap] = density_surv
    record["density_repr"][n_snap] = density_repr
    record["fitness"][n_snap] = fitness
    record["n1"][n_snap] = n1
    record["s1"][n_snap] = s1
    record["entropy"][n_snap] = entropy
    record["surv_junk"][n_snap] = surv_junk
    record["repr_junk"][n_snap] = repr_junk
    record["fitness_junk"][n_snap] = fitness_junk

    return record

def run_output(n_run, population, record):
    pop_file = open("run_"+str(n_run)+"_pop.txt", "wb")
    rec_file = open("run_"+str(n_run)+"_rec.txt", "wb")
    try:
        cPickle.dump(population, pop_file)
        cPickle.dump(record, rec_file)
    finally:
        pop_file.close()
        rec_file.close()
