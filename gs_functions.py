###############
## LIBRARIES ##
###############

import scipy.stats
import numpy as np
from random import randint,sample
from copy import copy,deepcopy
import time
import os
from importlib import import_module
import sys
import cPickle
import warnings

#############################
## CONFIGURATION FUNCTIONS ##
#############################

def get_dir(dir_name):
    """Change to specified working directory and update PATH."""
    sys.path.remove(os.getcwd())
    try:
        os.chdir(dir_name)
    except OSError:
        loop = True
        q = raw_input(
                "Given directory does not exist. Create it? (y/n) ")
        while loop:
            if q in ["yes", "y", "YES", "Y", "Yes"]:
                os.mkdir(dir_name)
                os.chdir(dir_name)
                loop = False
            elif q in ["no", "n", "NO", "N", "No", ""]:
                exit("Aborting: no valid working directory given.")
            else:
                q = raw_input("Invalid input. Create directory? (y/n) ")
    sys.path.append(os.getcwd())
    print "Working directory: "+os.getcwd()

def get_conf(file_name):
    """Import specified configuration file for simulation."""
    try:
        p = import_module(file_name)
    except ImportError:
        print "No such file in simulation directory: " + file_name
        q = raw_input(
                "Enter correct config file name, or skip to abort: ")
        if q == "":
            exit("Aborting: no valid configuration file given.")
        else: 
            return get_conf(q)
    print "Config file: "+ file_name +".py"
    return p

def get_startpop(seed_name):
    """Import any seed population (or return blank)."""
    if seed_name == "": 
        print "Seed: None."
        return ""
    try:
        # Make sure includes extension
        seed_split = seed_name.split(".")
        if seed_split[-1] != "txt":
            seed_name = seed_name + ".txt"
        popfile = open(seed_name, "rb")
        poparray = cPickle.load(popfile) # Import population array
    except IOError:
        print "No such seed file: " + seed_name
        q = raw_input(
                "Enter correct path to seed file, or skip to abort: ")
        if q == "":
            exit("Aborting: no valid seed file given.")
        else:
            return get_startpop(q)
    print "Seed population file: " + seed_name
    return poparray

##############################
## RANDOM NUMBER GENERATION ##
##############################

rand = scipy.stats.uniform(0,1) # Uniform random number generator

def chance(z,n=1):
    """Generate array of independent booleans with P(True)=z."""
    return rand.rvs(n)<z

###########################
## POPULATION GENERATION ##
###########################

def make_chromosome(variance, n_base, chr_len, gen_map, s_dist, r_dist):
    """ Returns chromosome array of specified length and composition."""
    ### Returns a binary array of length n, with the proportion of 1's
    ### determined by the initial distribution specified (random or
    ### a constant percentage).
    variance = max(0, variance) # No negative variance values.
    sd = variance**0.5
    p=scipy.stats.truncnorm(-0.5/sd, 0.5/sd, loc=0.5, scale=sd).rvs(1) 
    # 0/1-truncated normal distribution with mean 0.5 and sd as given
    chromosome = (rand.rvs(chr_len)<p)*1
    # If survival and/or reproduction probability is not random, 
    # replace appropriate loci in genome with new 1/0 distribution.
    if s_dist != "random":
        s_loci = np.nonzero(gen_map<100)[0]
        s_pos = [range(n_base) + x for x in s_loci*10]
        chromosome[s_pos] = chance(s_dist, len(s_pos))
    if r_dist != "random":
        r_loci = np.nonzero(np.logical_and(gen_map>100,gen_map<200))[0]
        r_pos = [range(n_base) + x for x in r_loci*10]
        chromosome[r_pos] = chance(r_dist, len(r_pos))
    return list(chromosome)

def make_individual(age_random, max_ls, maturity, variance, n_base, 
        chr_len, gen_map, s_dist, r_dist):
    """ Generate array for an individual, with age and 2 chromosomes."""
    age = randint(0,max_ls-1) if age_random else (maturity-1)
    chromosomes = [make_chromosome(variance, n_base, chr_len, gen_map, 
            s_dist, r_dist) for _ in range(2)]
    return np.concatenate(([age], chromosomes[0], chromosomes[1]))

def make_population(start_pop, age_random, max_ls, maturity,  variance, 
        n_base, chr_len, gen_map, s_dist, r_dist):
    """ Generate starting population of given size and composition."""
    print "Generating starting population...",
    population = np.array([make_individual(age_random, max_ls, 
        maturity, variance, n_base, chr_len, gen_map, s_dist, 
        r_dist) for _ in range(start_pop)])
    print "done."
    return population

######################
## UPDATE FUNCTIONS ##
######################

def update_resources(res0, N, R, V, limit, verbose=False):
    """Implement consumption and regrowth of resources."""
    if verbose: print "Updating resources...",
    k = 1 if N>res0 else V
    res1 = int((res0-N)*k+R)
    res1 = min(max(res1, 0), limit)
    # Resources can't be negative or exceed limit.
    if verbose: print "Done. "+str(res0)+" -> "+str(res1)
    return res1

def get_subpop(population, gen_map, min_age, max_age, offset, n_base,
        chr_len, val_range):
    """Randomly select a subset of the population based on genotype."""
    subpop = np.empty(0,int)
    ages = population[:,0]
    for age in range(min_age, min(max_age, max(ages))):
        # Get indices of appropriate locus for that age:
        locus = np.nonzero(gen_map==(age+offset))[0][0]
        pos = np.arange(locus*n_base, (locus+1)*n_base)+1
        # Get sub-population of correct age and subset genome to locus:
        match = (ages == age)
        which = np.nonzero(match)[0]
        pop = population[match][:,np.append(pos, pos+chr_len)]
        # Get sum of genotype for every individual:
        gen = np.sum(pop, axis=1)
        # Convert genotypes into inclusion probabilities:
        inc_rates = val_range[gen]
        included = which[chance(inc_rates, len(inc_rates))]
        subpop = np.append(subpop, included)
    return subpop


def generate_children(sexual, population, parents, chr_len, r_rate): 
    """Generate array of children through a/sexual reproduction."""
    if not sexual:
	children = np.copy(population[parents])
    else:
        # Must be even number of parents:
        if len(parents)%2 != 0: parents = parents[:-1]
        # Randomly assign mating partners:
        np.random.shuffle(parents) 
        # Recombination between chromosomes in each parent:
        rr = np.copy(population[parents])
        chr1 = np.arange(chr_len)+1
        chr2 = chr1 + chr_len
        for n in rr:
            r_sites = sample(range(chr_len), int(chr_len*r_rate))
            for r in r_sites:
                swap = np.copy(n[chr1][r:])
                n[chr1][r:] = np.copy(n[chr2][r:])
                n[chr2][r:] = swap
        # Generate children from randomly-combined parental chromatids
        chr_choice = np.random.choice(["chr1","chr2"], len(rr))
        chr_dict = {"chr1":chr1, "chr2":chr2}
        for m in range(len(rr)/2):
            rr[2*m][chr_dict[chr_choice[2*m]]] = \
                    rr[2*m+1][chr_dict[chr_choice[2*m+1]]]
        children = np.copy(rr[::2])
    return(children)

def reproduction(population, maturity, max_ls, gen_map, n_base, chr_len, 
        r_range, m_rate, r_rate, sexual=False, verbose=False):
    """Select parents, generate children, mutate and add to population."""
    if verbose: print "Calculating reproduction...",
    parents = get_subpop(population, gen_map, maturity, max_ls, 100, 
            n_base, chr_len, r_range) # Select parents
    children = generate_children(sexual, population, parents, chr_len,
            r_rate) # Generate children from parents
    # Mutate children:
    children[children==0]=chance(m_rate,np.sum(children==0))
    children[children==1]=1-chance(0.1*m_rate, np.sum(children==1))
    children[:,0] = 0 # Make newborn
    population=np.vstack([population,children]) # Add to population
    if verbose:
        print "done. "+str(len(children))+" new individuals born."
    return(population)

def death(population, max_ls, gen_map, n_base, chr_len, d_range, 
        starvation_factor, verbose=False):
    """Select survivors and kill rest of population."""
    if verbose: print "Calculating death...",
    survivors = get_subpop(population, gen_map, 0, max_ls, 0, n_base,
        chr_len, 1-(d_range*starvation_factor))
    new_population = population[survivors]
    if verbose:
        dead = len(population) - len(survivors)
        print "done. "+str(dead)+" individuals died."
    return(new_population)

####################################
## RECORDING AND OUTPUT FUNCTIONS ##
####################################

def initialise_record(snapshot_stages, n_stages, max_ls, n_bases, 
        gen_map, chr_len, d_range, r_range, window_size):
    """ Create a new dictionary object for recording output data."""
    m = len(snapshot_stages)
    array1 = np.zeros([m,max_ls])
    array2 = np.zeros([m,2*n_bases+1])
    array3 = np.zeros(m)
    array4 = np.zeros(n_stages)
    record = {
        "gen_map":gen_map,
        "chr_len":chr_len,
        "d_range":d_range,
        "r_range":r_range,
        "snapshot_stages":snapshot_stages,
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
    """Record only population size, resource and starvation data."""
    record["population_size"][n_stage] = N
    record["resources"][n_stage] = resources
    record["starvation_factor"][n_stage] = x
    return(record)

def x_bincount(array):
    """Auxiliary bincount function for update_record."""
    return np.bincount(array, minlength=3)

def moving_average(a, n):
    """Calculate moving average of an array along a sliding window."""
    c = np.cumsum(a, dtype=float)
    c[n:] = c[n:]-c[:-n]
    return c[(n-1):]/n

def update_record(record, population, N, resources, x, gen_map, chr_len, 
        n_bases, d_range, r_range, maturity, max_ls, window_size, 
        n_stage, n_snap):
    """Record detailed population data at current stage."""
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
	if len(pop) > 0:
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
	else:
	    surv_mean[age] = 0
	    surv_sd[age] = 0
	    if age >= maturity:
	        repr_mean[age] = 0
		repr_sd[age] = 0
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
    gen = population[:,1:]
    p1 = np.sum(gen)/float(np.size(gen))
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
    try:
        record["age_distribution"][n_snap] = np.bincount(ages, 
            minlength = max_ls)/float(N)
    except ValueError:
        print str(max(ages))
        raise ValueError
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
    """Save population and record objects as output files."""
    print "Saving output files...", 
    pop_file = open("run_"+str(n_run)+"_pop.txt", "wb")
    rec_file = open("run_"+str(n_run)+"_rec.txt", "wb")
    try:
        cPickle.dump(population, pop_file)
        cPickle.dump(record, rec_file)
    finally:
        pop_file.close()
        rec_file.close()
        print "done."
