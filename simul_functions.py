import scipy.stats
import numpy as np
from random import randint
from copy import copy,deepcopy
import time
import os
from importlib import import_module
import sys

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
