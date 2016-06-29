###############
## LIBRARIES ##
###############

import scipy.stats, datetime, os, sys
import numpy as np
from random import randint,sample
from importlib import import_module
import cPickle as pickle

#############################
## CONFIGURATION FUNCTIONS ##
#############################

def get_dir(dir_name):
    """Change to specified working directory and update PATH."""
    try:
        sys.path.remove(os.getcwd())
        os.chdir(dir_name)
        sys.path = [os.getcwd()] + sys.path
    except OSError:
        exit("Error: Specified simulation directory does not exist.")

def gen_conf(c):
    """Generate derived configuration parameters from simple ones and
    add to configuration object."""
    c.g_dist = {
            "s":c.g_dist_s, # Survival
            "r":c.g_dist_r, # Reproduction
            "n":c.g_dist_n # Neutral
            }
    # Genome map: survival (0 to max), reproduction (maturity to max), neutral
    c.genmap = np.asarray(range(0,c.max_ls)+range(c.maturity+100,
        c.max_ls+100)+range(200, 200+c.n_neutral))
    if c.sexual: c.repr_bound[1] *= 2
    c.chr_len = len(c.genmap) * c.n_base # Length of chromosome in binary units
    c.d_range = np.linspace(c.death_bound[1], c.death_bound[0],
            2*c.n_base+1) # max to min death rate
    c.r_range = np.linspace(c.repr_bound[0],c.repr_bound[1],
            2*c.n_base+1) # min to max repr rate
    if type(c.number_of_snapshots) is float:
        c.number_of_snapshots = int(c.number_of_snapshots * c.number_of_stages)
    c.snapshot_stages = np.around(np.linspace(0,c.number_of_stages-1,
            c.number_of_snapshots),0) 
    c.params = {"sexual":c.sexual, "chr_len":c.chr_len, "n_base":c.n_base,
        "maturity":c.maturity, "max_ls":c.max_ls, "age_random":c.age_random,
        "start_pop":c.start_pop, "g_dist":c.g_dist}
    # Dictionary for population initialisation
    return(c)

def get_conf(file_name):
    """Import specified configuration file for simulation."""
    try:
        p = gen_conf(import_module(file_name))
    except ImportError:
        print "No such file in simulation directory: " + file_name
        q = raw_input(
                "Enter correct config file name, or skip to abort: ")
        if q == "":
            exit("Aborting: no valid configuration file given.")
        else:
            return get_conf(q)
    logprint("Config file: "+ file_name +".py")
    return p

def get_startpop(seed_name):
    """Import any seed population (or return blank)."""
    if seed_name == "":
        logprint("Seed: None.")
        return ""
    try:
        # Make sure includes extension
        seed_split = seed_name.split(".")
        if seed_split[-1] != "txt":
            seed_name = seed_name + ".txt"
        popfile = open(seed_name, "rb")
        poparray = pickle.load(popfile) # Import population array
    except IOError:
        print "No such seed file: " + seed_name
        q = raw_input(
                "Enter correct path to seed file, or skip to abort: ")
        if q == "":
            exit("Aborting: no valid seed file given.")
        else:
            return get_startpop(q)
    logprint("Seed population file: " + seed_name)
    return poparray

##############################
## RANDOM NUMBER GENERATION ##
##############################

rand = scipy.stats.uniform(0,1) # Uniform random number generator

def chance(p,n=1):
    """Generate array of (n or shape n=(x,y)) independent booleans with P(True)=z."""
    return rand.rvs(n)<p

###########################
## POPULATION GENERATION ##
###########################

def make_genome_array(start_pop, chr_len, genmap, n_base, g_dist):
    """
    Generate genomes for start_pop individuals with chromosome length of chr_len
    bits and locus length of n_base bits. Set the genome array values (distribution
    of 1's and 0's) according to genmap and g_dist.
    """
    # Initialise genome array:
    genome_array = np.zeros([start_pop,chr_len*2])
    # Use genome map to determine probability distribution for each locus:
    loci = {
        "s":np.nonzero(genmap<100)[0],
        "r":np.nonzero(np.logical_and(genmap>=100,genmap<200))[0],
        "n":np.nonzero(genmap>=200)[0]
        }
    # Set genome array values according to given probabilities:
    for k in loci.keys():
        pos = np.array([range(n_base) + x for x in loci[k]*10])
        pos = np.append(pos, pos + chr_len)
        genome_array[:, pos] = chance(g_dist[k], [start_pop, len(pos)])
    return genome_array.astype("int")

######################
## UPDATE FUNCTIONS ##
######################

def update_resources(res0, N, R, V, limit, verbose=False):
    """Implement consumption and regrowth of resources depending on 
    population size (under the variable-resource condition)."""
    if verbose: print "Updating resources...",
    k = 1 if N>res0 else V
    res1 = int((res0-N)*k+R) # This means individuals can consume future resources?
    res1 = min(max(res1, 0), limit)
    # Resources can't be negative or exceed limit.
    if verbose: logprint("Done. "+str(res0)+" -> "+str(res1))
    return res1

####################################
## RECORDING AND OUTPUT FUNCTIONS ##
####################################

def run_output(n_run, population, record, window_size):
    """Save population and record objects as output files."""
    record.final_update(n_run, window_size)
    # Save output files
    logprint("Saving output files...", False)
    pop_file = open("run_"+str(n_run)+"_pop.txt", "wb")
    rec_file = open("run_"+str(n_run)+"_rec.txt", "wb")
    try:
        pickle.dump(population, pop_file)
        pickle.dump(record.record, rec_file)
    finally:
        pop_file.close()
        rec_file.close()
        logprint("done.")

def print_runtime(starttime, endtime):
    runtime = endtime - starttime
    days = runtime.days
    hours = runtime.seconds/3600
    minutes = runtime.seconds/60 - hours*60
    seconds = runtime.seconds - minutes*60 - hours*3600
    logprint("Total runtime :", False)
    if days != 0:
        logprint("{d} days".format(d=days)+", ", False)
    if hours != 0:
        logprint("{h} hours".format(h=hours)+", ", False)
    if minutes != 0:
        logprint("{m} minutes".format(m=minutes)+", ", False)
    logprint("{s} seconds".format(s=seconds)+".\n")

def logprint(string, newline=True, logfile="log.txt"):
    """Print string to both stdout and lof file."""
    print string if newline else string,
    if logfile != "":
        log = open(logfile, "a")
	log.write(string)
	if newline: log.write("\n")
	log.close()
