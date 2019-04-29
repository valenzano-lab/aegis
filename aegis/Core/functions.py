import datetime, warnings, numpy as np, scipy.stats as st, copy, itertools
from dateutil.relativedelta import relativedelta as delta

###################
## Randomisation ##
###################

def chance(p,n=1,prng=0):
    """Generate array (of shape specified by n, where n is either an integer
    or a tuple of integers) of independent booleans with P(True)=z."""
    if prng==0: prng = np.random.RandomState()
    return prng.random_sample(n) < p

###########################
## Config Initialisation ##
###########################

def make_windows(array, ws, last=True):
    """For an array of integers, for every integer, return range(ws) around
    that integer. for the first element of array expand only forward, for the
    last expand only backward."""
    if ws<1:
        s = "Window size must be greater than zero."
        raise ValueError(s)
    elif ws == 1:
        return np.array(array).reshape(len(array),1)
    else:
        ws = ws/2*2
        res = [range(ws)]
        for i in array[1:-1]:
            res.append(range(i-ws/2, i+ws/2))
        if last: res.append(range(array[-1]-ws, array[-1]))
        else: res.append(range(array[-1]-ws/2, array[-1]+ws/2))
        return np.array(res)

def make_mapping(bitstring_length, n_states):
    """Creates a mapping from all possible bit strings of length bitstring_lenght
    to the set range(n_states) such that all bit strings are mapped to their sum.
    Return a numpy array such that the index represents the integer the bit string
    codes for in binary and the value represents the sum of that bit string."""
    mapping = np.zeros(2**bitstring_length).astype(int)
    genotypes = np.array(map(list, itertools.product([0,1],\
            repeat=bitstring_length))).sum(1)
    for i in xrange(n_states):
        ix = np.where(genotypes==i)
        mapping[ix] = i
    return mapping

###############################
## Population Initialisation ##
###############################

def init_ages():
    """Return an array specifying that a new age vector should be generated
    during Population initialisation."""
    return np.array([-1])

def init_genomes():
    """Return an array specifying that a new genome array should be generated
    during Population initialisation."""
    return np.array([[-1],[-1]])

def init_generations():
    """Return an array specifying that a new generation vector should be
    generated during Population initialisation."""
    return np.array([-1])

def init_gentimes():
    """Return an array specifying that a new generation-time vector should be
    generated during Population initialisation."""
    return np.array([-1])

##########
## Time ##
##########

def timenow(readable=True, fstring='on %Y-%m-%d at %X'):
    """Save the current time as a datetime object or a human-readable
    string."""
    dt = datetime.datetime.now()
    if not readable: return dt
    return dt.strftime(fstring)

def timediff(starttime, endtime):
    """Find the time difference between two datetime objects and return
    as a human-readable string."""
    time, outstr = delta(endtime, starttime), ""
    units = ["days", "hours", "minutes", "seconds"]
    values = np.array([getattr(time, unit) for unit in units])
    after = [", ", ", ", " and ", ""]
    for n in xrange(len(units)):
        g = getattr(time, units[n])
        report = (g != 0) or (units[n]=="seconds")
        if report: outstr += "{0} {1}{2}".format(g, units[n], after[n])
    return outstr

def get_runtime(starttime, endtime, prefix = "Total runtime"):
    """Compute the runtime of a process from the start and end times
    and print the output as a human-readable string."""
    runtime = timediff(starttime, endtime)
    return "{}: {}.".format(prefix, runtime)

###########################
## Five-number Summaries ##
###########################

def quantile(array, p, interpolation="linear"):
    """Get the p-quantile of a numeric array using np.percentile."""
    return np.percentile(array, p*100, interpolation=interpolation)

def fivenum(array):
    """Return the five-number summary of a numeric array."""
    return np.array([quantile(array, p) for p in np.arange(0, 1.1, 0.25)])

#################################################
## Comparing Dictionaries with Compound Values ##
#################################################

def deep_key(key, dict1, dict2, exact=True, prng_id=True):
    """Compare the values linked to a given key in two dictionaries
    according to the types of those values."""
    print key, # for debugging
    v1, v2 = dict1[key], dict2[key]
    f = np.allclose if not exact else np.array_equal
    if key=="prng":
        if prng_id: # check identical objects (not independent)
            sc_prng = copy.copy(v1)
            return f(v2.rand(100), sc_prng.rand(100)) and \
                    f(v1.rand(100), sc_prng.rand(100)) and not \
                    f(v1.rand(100), v2.rand(100))
        else: # check same but independent
            sc_prng1 = copy.copy(v1)
            check1 = f(v2.rand(100), sc_prng1.rand(100))
            sc_prng2 = copy.copy(v1)
            check2 = not f(sc_prng1.rand(100), sc_prng2.rand(100))
            return check1 and check2
    elif type(v1) is not type(v2): return False
    elif callable(v1):
        warnings.warn("Cannot compare callable values.", UserWarning)
        return True
    elif isinstance(v1, dict): return deep_eq(v1, v2)
    elif isinstance(v1, np.ndarray): return f(v1,v2)
    elif isinstance(v1, tuple): # state of numpy prng
        for w1,w2 in zip(v1,v2):
            if isinstance(w1, np.ndarray): return f(w1,w2)
            else: return w1==w2
    else: return v1 == v2

def deep_eq(d1, d2, exact=True, prng_id=True):
    """Compare two dictionaries element-wise according to the types of
    their constituent values."""
    if sorted(d1.keys()) != sorted(d2.keys()): return False
    for k in d1.keys():
        if not deep_key(k,d1,d2,exact,prng_id): return False
    return True

###################################
# Correcting Recombination Rates ##
###################################

def correct_r_rate(r_rate):
    """Correct a per-bit crossover rate parameter value to account for
    forward and reverse crossovers occurring at the same position"""
    if r_rate < 0 or r_rate > 1:
        raise ValueError("Invalid crossover rate: must be between 0 and 1.")
    r = r_rate if r_rate <= 0.5 else (1-r_rate)
    # R = P[recombination at site] = P[forward and no backward recombination or backward and no forward recombination] = 2 * (r/2) * (1-r/2)
    # this yields the result when solved for R with constraint 0<=R<=1
    return 1-(1-2*r)**0.5
