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
    try:
        sys.path.remove(os.getcwd())
        os.chdir(dir_name)
        sys.path.append(os.getcwd())
    except OSError:
        exit("Error: Specified simulation directory does not exist.")

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

def run_output(n_run, population, record, logfile, window_size):
    """Save population and record objects as output files."""
    record.final_update(n_run, window_size)
    # Save output files
    logprint("Saving output files...", logfile, False)
    pop_file = open("run_"+str(n_run)+"_pop.txt", "wb")
    rec_file = open("run_"+str(n_run)+"_rec.txt", "wb")
    try:
        cPickle.dump(population, pop_file)
        cPickle.dump(record.record, rec_file)
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
