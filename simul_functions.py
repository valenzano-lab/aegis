import scipy.stats

rand = scipy.stats.uniform(0,1) # Generate random number generator

def chance(z):
    return rand.rvs(1)<z

def starting_genome(var,n,gen_map,s_dist,r_dist):
    ### Returns a binary array of length n, with the proportion of 1's
    ### determined by the initial distribution specified (random or
    ### a constant percentage).
    var=min(1.4, var)
    sd = var**0.5
    p=scipy.stats.truncnorm(-0.5/sd, 0.5/sd, loc=0.5, scale=sd).rvs(1) 
    # 0/1-truncated normal distribution with mean 0.5 and sd as given
    a = (rand.rvs(n)<p)*1
    # If survival and/or reproduction probability is not random, 
    # replace appropriate loci in genome with new 1/0 distribution.
    if s_dist != "random":
        s = float(s_dist)/100
        s_loci = np.nonzero(gen_map<100)
        a[s_loci] = rand.rvs(len(s_loci))<s
    if r_dist != "random":
        r = float(r_dist)/100
        r_loci = np.nonzero(np.logical_and(gen_map>100,gen_map<200))
        a[r_loci] = rand.rvs(len(r_loci))<r
    return a

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def my_shuffle(array):
    shuffle(array)
    return array

