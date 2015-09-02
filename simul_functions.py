import scipy.stats

rand = scipy.stats.uniform(0,1) # Generate random number generator

def chance(z):
    return rand.rvs(1)<=z

def starting_genome(var,n):
    ### Returns a binary array of length n, with the proportion of 1's
    ### determined by a truncated normal distribution with variance var
    var=min(1.4, var)
    sd = var**0.5
    p=scipy.stats.truncnorm(-0.5/sd, 0.5/sd, loc=0.5, scale=sd).rvs(1) 
    # Normal distribution with mean 0.5 and sd as given, truncated to between 0 and 1.
    a = (rand.rvs(n)<=p)*1
    return a

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def my_shuffle(array):
    shuffle(array)
    return array

