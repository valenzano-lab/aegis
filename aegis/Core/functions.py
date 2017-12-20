import scipy.stats as st
import random
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta as delta
#import numba

###################
## Randomisation ##
###################

#@numba.jit(nopython=True)
def chance(p,n=1):
    """Generate array (of shape specified by n, where n is either an integer
    or a tuple of integers) of independent booleans with P(True)=z."""
    return np.random.random(n) < p

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
        report = (g != 0)
        if report: outstr += "{0} {1}{2}".format(g, unit, after[n])
    return outstr
# TODO: Test this

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
