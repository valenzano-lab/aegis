###############
## FUNCTIONS ##
###############

def chance(z):
    z = round(z * 1000000, 0)
    if (randint(1, 1000000) <= z):
        y = True
    else:
        y = False
    return y

def starting_genome(var,n):
    ### Returns a binary array of length n, with the proportion of 1's
    ### determined by a truncated normal distribution with variance var
    var=min(1.4, var)
    sd = var**0.5
    p=stats.truncnorm(-0.5/sd, 0.5/sd, loc=0.5, scale=sd).rvs(1) 
    # Normal distribution with mean 0.5 and sd as given, truncated to between 0 and 1.
    a = (stats.uniform(0,1).rvs(n)<p)*1
    return a

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def my_shuffle(array):
    shuffle(array)
    return array

