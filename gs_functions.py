###############
## LIBRARIES ##
###############

import scipy.stats
import numpy as np
from random import randint,sample
import datetime
import os
from importlib import import_module
import sys
import cPickle

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

def get_conf(file_name, logfile):
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
    logprint("Config file: "+ file_name +".py", logfile)
    return p

def get_startpop(seed_name, logfile):
    """Import any seed population (or return blank)."""
    if seed_name == "": 
        logprint("Seed: None.", logfile)
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
    logprint("Seed population file: " + seed_name, logfile)
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

def make_genome_array(start_pop, sd, n_base, chr_len, gen_map, 
        s_dist, r_dist):
    trunc = scipy.stats.truncnorm(-0.5/sd, 0.5/sd, loc=0.5, scale=sd)
    p = trunc.rvs(start_pop*2)
    genome_array = np.random.uniform(size=[start_pop, chr_len*2])
    genome_array[:, :chr_len] = np.apply_along_axis(
            lambda x: x<p[:start_pop], 0, genome_array[:, :chr_len])
    genome_array[:, chr_len:] = np.apply_along_axis(
            lambda x: x<p[start_pop:], 0, genome_array[:, chr_len:])
    genome_array = genome_array.astype("bool")
    if s_dist != "random":
        s_loci = np.nonzero(gen_map<100)[0]
        s_pos = np.array([range(n_base) + x for x in s_loci*10])
        s_pos = np.append(s_pos, s_pos + chr_len)
        genome_array[:, s_pos] = chance(s_dist, [start_pop, len(s_pos)])
    if r_dist != "random":
        r_loci = np.nonzero(np.logical_and(gen_map>100,gen_map<200))[0]
        r_pos = np.array([range(n_base) + x for x in r_loci*10])
        r_pos = np.append(r_pos, r_pos + chr_len)
        genome_array[:, r_pos] = chance(r_dist, [start_pop, len(r_pos)])
    return genome_array.astype("int")

######################
## UPDATE FUNCTIONS ##
######################

def update_resources(res0, N, R, V, limit, logfile, verbose=False):
    """Implement consumption and regrowth of resources."""
    if verbose: print "Updating resources...",
    k = 1 if N>res0 else V
    res1 = int((res0-N)*k+R)
    res1 = min(max(res1, 0), limit)
    # Resources can't be negative or exceed limit.
    if verbose: logprint("Done. "+str(res0)+" -> "+str(res1), logfile)
    return res1

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
        "chr_len":np.array([chr_len]),
        "d_range":d_range,
        "r_range":r_range,
        "snapshot_stages":snapshot_stages+1,
        "population_size":np.copy(array4),
        "resources":np.copy(array4),
        "starvation_factor":np.copy(array4),
        "age_distribution":np.copy(array1),
    	"death_mean":np.copy(array1),
    	"death_sd":np.copy(array1),
    	"repr_mean":np.copy(array1),
    	"repr_sd":np.copy(array1),
    	"density_surv":np.copy(array2),
    	"density_repr":np.copy(array2),
    	"n1":np.zeros([m,chr_len]),
    	"s1":np.zeros([m,chr_len-window_size+1]),
    	"fitness":np.copy(array1),
    	"entropy":np.copy(array3),
    	"junk_death":np.copy(array3),
    	"junk_repr":np.copy(array3),
    	"junk_fitness":np.copy(array3)
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

def rolling_window(a, window):
    """Efficiently compute a rolling window for a numpy array."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strd = a.strides + (a.strides[-1],) # strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strd)

def update_record(record, population, resources, x, gen_map, chr_len, 
        n_bases, d_range, r_range, maturity, max_ls, n_stage, n_snap):
    """Record detailed population data at current stage."""
    N = population.N
    ages = population.ages
    genomes = population.genomes
    record = quick_update(record, n_stage, N, resources, x)
    b = n_bases # Binary units per locus

    ## AGE-DEPENDENT STATS ##
    # Genotype sum distributions:
    density_surv = np.zeros((2*b+1,))
    density_repr = np.zeros((2*b+1,))
    # Mean death/repr rates by age:
    death_mean = np.zeros(max_ls)
    repr_mean = np.zeros(max_ls)
    # Death/repr rate SD by age:
    death_sd = np.zeros(max_ls)
    repr_sd = np.zeros(max_ls)
    # Loop over ages:
    for age in range(max(ages)):
        pop = genomes[ages==age]
	if len(pop) > 0:
            # Find loci and binary units:
            try:
                surv_locus = np.nonzero(gen_map==age)[0][0]
            except:
                print max(ages)
            surv_pos = np.arange(surv_locus*b, (surv_locus+1)*b)
            # Subset array to relevant columns and find genotypes:
            surv_pop = pop[:,np.append(surv_pos, surv_pos+chr_len)]
            surv_gen = np.sum(surv_pop, axis=1)
            # Find death/reproduction rates:
            death_rates = d_range[surv_gen]
            # Calculate statistics:
            death_mean[age] = np.mean(death_rates)
            death_sd[age] = np.std(death_rates)
            density_surv += np.bincount(surv_gen, minlength=2*b+1)
            if age>=maturity:
                # Same for reproduction if they're adults
                repr_locus = np.nonzero(gen_map==(age+100))[0][0]
                repr_pos = np.arange(repr_locus*b, (repr_locus+1)*b)
                repr_pop = pop[:,np.append(repr_pos, repr_pos+chr_len)]
                repr_gen = np.sum(repr_pop, axis=1)
                repr_rates = r_range[repr_gen]
                repr_mean[age] = np.mean(repr_rates)
                repr_sd[age] = np.std(repr_rates)
                density_repr += np.bincount(repr_gen, minlength=2*b+1)
	else:
	    death_mean[age] = 0
	    death_sd[age] = 0
	    if age >= maturity:
	        repr_mean[age] = 0
		repr_sd[age] = 0
    # Average densities over whole genomes
    density_surv /= float(N)
    density_repr /= float(N)

    ## AGE-INVARIANT STATS ##
    # Frequency of 1's at each position on chromosome:
    # Average over array, then average at each position over chromosomes
    n1s = np.sum(genomes, 0)/float(N)
    n1 = (n1s[np.arange(chr_len)]+n1s[np.arange(chr_len)+chr_len])/2 
    # Shannon-Weaver entropy over entire genome genomes
    gen = genomes[:,1:]
    p1 = np.sum(gen)/float(np.size(gen))
    entropy = scipy.stats.entropy(np.array([1-p1, p1]))
    # Junk stats calculated from neutral locus
    neut_locus = np.nonzero(gen_map==201)[0][0] 
    neut_pos = np.arange(neut_locus*b, (neut_locus+1)*b)
    neut_pop = genomes[:,np.append(neut_pos, neut_pos+chr_len)]
    neut_gen = np.sum(neut_pop, axis=1)
    junk_death = np.mean(d_range[neut_gen])
    junk_repr = np.mean(r_range[neut_gen]) # Junk SDs?

    ## APPEND RECORD OBJECT ##
    record["age_distribution"][n_snap] = np.bincount(ages, 
        minlength = max_ls)/float(N)
    record["death_mean"][n_snap] = death_mean
    record["death_sd"][n_snap] = death_sd
    record["repr_mean"][n_snap] = repr_mean
    record["repr_sd"][n_snap] = repr_sd
    record["density_surv"][n_snap] = density_surv
    record["density_repr"][n_snap] = density_repr
    record["n1"][n_snap] = n1
    record["entropy"][n_snap] = entropy
    record["junk_death"][n_snap] = junk_death
    record["junk_repr"][n_snap] = junk_repr

    return record

def run_output(n_run, population, record, logfile, window_size):
    """Save population and record objects as output files."""
    # Perform whole-array computations of fitness and rolling STD
    record["s1"] = np.std(rolling_window(record["n1"], window_size),2)
    x_surv = np.cumprod(1-record["death_mean"],1)
    record["fitness"] = np.cumsum(x_surv * record["repr_mean"],1)
    record["junk_fitness"] = (1-record["junk_death"])*record["junk_repr"]
    # Save output files
    logprint("Saving output files...", logfile, False)
    pop_file = open("run_"+str(n_run)+"_pop.txt", "wb")
    rec_file = open("run_"+str(n_run)+"_rec.txt", "wb")
    try:
        cPickle.dump(population, pop_file)
        cPickle.dump(record, rec_file)
    finally:
        pop_file.close()
        rec_file.close()
        logprint("done.", logfile)

def print_runtime(starttime, endtime, logfile):
    runtime = endtime - starttime
    days = runtime.days
    hours = runtime.seconds/3600
    minutes = runtime.seconds/60 - hours*60
    seconds = runtime.seconds - minutes*60 - hours*3600
    logprint("Total runtime :", logfile, False)
    if days != 0: 
        logprint("{d} days".format(d=days)+", ", logfile, False)
    if hours != 0: 
        logprint("{h} hours".format(h=hours)+", ", logfile, False)
    if minutes != 0: 
        logprint("{m} minutes".format(m=minutes)+", ", logfile, False)
    logprint("{s} seconds".format(s=seconds)+".\n", logfile)

def logprint(string, logfile, newline=True):
    if newline:
        print string
        if logfile != "": logfile.write(string+"\n")
    else:
        print string,
        if logfile != "": logfile.write(string)

