import scipy.stats
import numpy as np
from random import randint

rand = scipy.stats.uniform(0,1) # Generate random number generator

def chance(z,n=1):
    return rand.rvs(n)<z

def starting_genome(var,n,gen_map,s_dist,r_dist):
    ### Returns a binary array of length n, with the proportion of 1's
    ### determined by the initial distribution specified (random or
    ### a constant percentage).
    if n % len(gen_map) != 0:
        raise ValueError("Invalid genome length; must be integer multiple of genome map.")
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
        a[s_pos] = rand.rvs(len(s_pos))<s
    if r_dist != "random":
        r = float(r_dist)/100
        r_loci = np.nonzero(np.logical_and(gen_map>100,gen_map<200))[0]
        a[r_loci] = rand.rvs(len(r_loci))<r
        r_pos = []
        for x in r_loci:
            r_pos.append(range(x*q, (x+1)*q))
        a[r_pos] = chance(s, len(r_pos))
    return a

def make_individual(age_var, var, n, gen_map, s_dist, r_dist):
    ### Returns a [1,2n+1]-dimensional array representing an individual
    ### with a starting age and two length-n chromosomes
    age = 15 if age_var=="y" else randint(0,70) # Uniform?
    chr1 = starting_genome(var, n, gen_map, s_dist, r_dist)
    chr2 = starting_genome(var, n, gen_map, s_dist, r_dist)
    individual = np.concatenate(([age], chr1, chr2))
    return individual
