########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Infodict, Config                                            #
# Description: Simulation configuration information imported from a    #
#   config file and saved as a dictionary-like object for use by other #
#   classes. The dictionary keys are linked to both values and         #
#   information entries for identifying the function of each entry.    #
########################################################################

## PACKAGE IMPORT ##
import numpy as np
import copy

## AUXILIARY FUNCTIONS ##

def deepeq(dict1, dict2, verbose=False):
    """Compare two dictionaries element-wise according to the types of
    their constituent values."""
    if sorted(dict1.keys()) != sorted(dict2.keys()): # First check keys
        if verbose: 
            print "Non-shared keys:"
            print set(dict1.keys()).difference(set(dict2.keys()))
            print set(dict2.keys()).difference(set(dict1.keys()))
        return False # First check keys
    for k in dict1.keys():
        if verbose: print k
        v1, v2 = dict1[k], dict2[k]
        if type(v1) is not type(v2): 
            if verbose: print "Type mismatch!"
            return False
        elif isinstance(v1, dict):
            if not deepeq(v1, v2): 
                if verbose: print v1, v2
                return False
        elif isinstance(v1, np.ndarray):
            if not np.allclose(v1, v2): 
                if verbose: print v1, v2
                return False
        elif v1 != v2: return False
    return True

## CLASS ##

class Infodict:
    """Dictionary-like object in which each element is associated to
    both a value and an information string describing its purpose,
    and in which selecting for lists/arrays of keys returns 
    corresponding lists/arrays of values."""

    def __init__(self):
        self.__valdict__ = {}
        self.__infdict__ = {}

    # Constructors

    def put(self, key, value, info):
        """Set a key's value and description simultaneously."""
        self.__valdict__[key] = value
        self.__infdict__[key] = info

    def __put_single__(self, key, value, mode):
        """Change the value/infostring of an existing key,
        preventing naive setting of new items wihout both parts."""
        indict = self.__valdict__ if mode == "val" else self.__infdict__
        if key in self.keys():
            indict[key] = value
        else:
            errstr = "Infodict keys must have both a value and an info-string;"
            errstr += " use .put() to specify both simultaneously."
            raise SyntaxError(errstr)

    def put_value(self, key, value):
        """Change the value of an existing key while preventing
        naive setting of new items without an infostring."""
        self.__put_single__(key, value, "val")

    def put_info(self, key, info):
        """Change the infostring of an existing key while preventing
        naive setting of new items without a value."""
        self.__put_single__(key, info, "inf")
            
    def __setitem__(self, key, value): self.put_value(key, value)

    # Single selectors

    def __get__(self, key, mode):
        """Get the value/infostring associated with a single key."""
        outdict = self.__valdict__ if mode == "val" else self.__infdict__
        return outdict[key]

    def get_value(self, key):
        """Get the value associated with a single key."""
        return self.__get__(key, "val")

    def get_info(self, key):
        """Get the infostring associated with a single key."""
        return self.__get__(key, "inf")

    # Multiple selectors

    def __gets__(self, keys, mode):
        """Get the values/infostrings of one or more keys, returning
        the result as a list or array if multiple keys are given."""
        if type(keys) in [list, np.ndarray, tuple] and len(keys) > 1:
            return [self.__get__(k, mode) for k in keys]
        elif type(keys) in [list, np.ndarray, tuple]:
            return self.__get__(keys[0], mode)
        else:
            return self.__get__(keys, mode)

    def get_values(self, keys):
        """Get the values of one or more keys, returning the result
        as a list or array if multiple keys are given."""
        return self.__gets__(keys, "val")

    def get_infos(self, keys):
        """Get the infostrings of one or more keys, returning the result
        as a list or array if multiple keys are given."""
        return self.__gets__(keys, "inf")

    def __getitem__(self, *keys): return self.get_values(keys[0])

    # Item deletion

    def delete_item(self, key):
        """Remove a key, along with its value and infostring."""
        del self.__valdict__[key], self.__infdict__[key]

    def __delitem__(self, key): self.delete_item(key)

    # Redirect basic dictionary methods
    def keys(self): return self.__infdict__.keys()
    def values(self): return self.__valdict__.values()
    def infos(self): return self.__infdict__.values()
    def has_key(self, key): return self.__infdict__.has_key(key)

    # Comparison and equality
    def eq_infs(self, other):
        if not isinstance(other, self.__class__): return NotImplemented
        return self.__infdict__ == other.__infdict__
    def eq_vals(self, other):
        if not isinstance(other, self.__class__): return NotImplemented
        return deepeq(self.__valdict__, other.__valdict__)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.eq_infs(other) and self.eq_vals(other)
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented
    def __hash__(self):
        return hash(tuple(sorted(self.__valdict__.items() + \
                self.__infdict__.items())))
    def copy(self): return copy.deepcopy(self)

    #! Finish this

    # Copy method

    def subdict(self, keys, asdict=True):
        """Return a dictionary or infodict containing a specified 
        subset of this object's keys."""
        valdict = {k: self.get_value(k) for k in keys}
        if asdict: return valdict
        infdict = {k: self.get_info(k) for k in keys}
        child = Infodict()
        child.__valdict__ = valdict
        child.__infdict__ = infdict
        return child 

class Config(Infodict):
    """Object derived from imported config module."""

    def __init__(self, c):
        """Import basic attributes from a config-file object and set data
        descriptors."""
        self.__valdict__ = {}
        self.__infdict__ = {}
        def mirror(name, infostring):
            """Set a data element's value from the attribute of the same name
            in c, while also setting its description."""
            self.put(name, getattr(c,name), infostring)
        ## Set data elements and information ##
        # Reproductive mode
        mirror("repr_mode", "The reproductive mode of the population:\
                'sexual' (recombination and assortment), 'recombine_only',\
                'assort_only', or 'asexual' (no recombination or assortment).\
                [str]") # Test that this is a valid value
        # Run parameters
        mirror("number_of_runs", "The number of runs to be \
            executed in parallel to generate summary data. [int]")
        mirror("number_of_stages", "The number of stages per run. [int]")
        mirror("number_of_snapshots", "The number of stages in the run\
                at which to save detailed data (evenly distributed). [int]")
        mirror("start_pop", "Population size at stage 0. [int]")
        mirror("max_fail", "Maximum number of failed attempts per run. [int]")
        # Death and reproduction
        mirror("death_bound", "Min and max death rates. [float array]")
        mirror("repr_bound", "Min and max reproduction rates.\
            [float array]")
        mirror("r_rate", "Per-bit recombination rate, if applicable. \
                [float]")
        mirror("m_rate", "Per-bit mutation rate during reproduction. \
                [float]")
        mirror("m_ratio", "The ratio of positive (0 -> 1) to negative\
                (1 -> 0) mutations during reproduction. Typically negative.\
                [float]")
        # Genome structure
        mirror("g_dist_s", "Initial proportion of 1's in survival loci\
                at stage 0. [float]")
        mirror("g_dist_r", "Initial proportion of 1's in reproduction\
                loci at stage 0. [float]")
        mirror("g_dist_n", "Initial proportion of 1's in neutral loci\
                at stage 0. [float]")
        mirror("n_base", "Number of bits per locus in the genome. [int]")
        mirror("n_neutral", "Number of neutral loci per genome;\
                more loci reduces the effect of linkage at any given one.\
                [int]")
        mirror("repr_offset", "Number by which to offset reproductive\
                loci in genome map (must > maximum lifespan). [int]")
        mirror("neut_offset", "Number by which to offset neutral loci\
                in genome map (must > repr_offset + max lifespan - maturity).\
                [int]")
        # Resources and starvation
        mirror("res_start", "The resource level at stage 0. [int]")
        mirror("res_var", "Whether resources vary through consumption\
                and regrowth (True) or remain constant (False). [bool]")
        mirror("res_limit", "Maximum resource level (-1 = infinite).\
                [int]") #!
        mirror("R", "Arithmetic regrowth increment under variable\
                resources. [int]")
        mirror("V", "Geometric regrowth rate under variable\
                resources. [float]")
        mirror("surv_pen", "Whether to penalise survival under \
                starvation by multiplying the death rate. [bool]")
        mirror("repr_pen", "Whether to penalise reproduction under \
                starvation by dividing the reproduction rate. [bool]")
        mirror("death_inc", "Factor for multiplying death rate of \
                individuals under starvation. [float]")
        mirror("repr_dec", "Factor for dividing reproduction rate of \
                individuals under starvation. [float]")
        # Life history
        mirror("max_ls", "Maximum lifespan for the population in stages\
                (-1 = indefinite). [int]") #!
        mirror("maturity", "Age of reproductive maturation in stages;\
                before this, P(reproduction) = 0. [int]")
        # Sliding windows
        mirror("windows", "Width of sliding windows for recording \
                along-genome variation in bit value and along-run \
                variation in population size and resource levels. [dict]")
        # Output mode
        mirror("output_prefix", "Prefix for output file names. [str]")
        mirror("output_mode", "Level of information to retain in simulation\
                output: 0 = records only, 1 = records + final population,\
                2 = records + all snapshot populations. [int]")
        self.check()

    # Check for invalid config construction
    def check(self):
        """Confirm that population parameters are compatible before
        starting simulation."""
        states = ['asexual','recombine_only','assort_only','sexual']
        if self["repr_mode"] not in states:
            s0 = "Invalid reproductive mode: '{}'.".format(self["repr_mode"])
            s1 = " Must be one of: {}.".format(", ".join(states))
            print self.get_values(["repr_mode", "repr_mode"])
            raise ValueError(s0 + s1)
        if self["maturity"] > self["max_ls"] - 2:
            s = "Age of maturity must be at least 2 less than max lifespan."
            raise ValueError(s)
        if self["repr_offset"] < self["max_ls"]:
            s = "Offset for reproductive loci must be >= max lifespan."
            raise ValueError(s)
        if self["neut_offset"] - self["repr_offset"] < self["max_ls"]:
            s = "Difference between offset values must be >= max lifespan."
            raise ValueError(s)
        return True

    # Generate derived attributes

    def generate(self):
        """Generate derived configuration attributes from simple ones and
        add to configuration object."""
        # Compute values
        self.check()
        # Genome structure
        g_dist_dict = { # Dictionary of initial proportion of 1's in genome loci
                "s":self["g_dist_s"], # Survival
                "r":self["g_dist_r"], # Reproduction
                "n":self["g_dist_n"] # Neutral
                }
        self.put("g_dist", g_dist_dict, "Dictionary of values specifying\
                initial proportion of 1's in each locus type (s/r/n) during\
                population initialisation. [dict of floats]")
        genmap = np.concatenate([
            np.arange(self["max_ls"]),
            np.arange(self["maturity"],self["max_ls"]) + self["repr_offset"],
            np.arange(self["n_neutral"]) + self["neut_offset"]], 0)
        self.put("genmap", genmap, "Genome map specifying the relative \
                position of each survival, reproduction and neutral locus \
                in the genome of each individual. [int array]")
        self.put("genmap_argsort", np.argsort(self["genmap"]), "Array specifying\
                how to sort the genome map (or the genome) to correctly order\
                the loci (survival 0-maxls, reproduction maturity-maxls, \
                neutral. [int array]")
        self.put("chr_len", len(self["genmap"]) * self["n_base"], "Length of one\
                chromosome in bits. [int]")
        self.put("n_states", 2*self["n_base"]+1, "Number of possible genotype \
                states (from 0 to 2*n_base). [int]")
        # Survival and reproduction
        self["death_bound"] = np.array(self["death_bound"])
        self["repr_bound"] = np.array(self["repr_bound"])
        if self["repr_mode"] in ["sexual", "assort_only"]: 
        # Double fertility in sexual case to control for relative
        # contribution to offspring
            self["repr_bound"] *= 2 
        self.put("surv_bound", 1-self["death_bound"][::-1], "Min and max \
                survival rates. [float array].")
        self.put("surv_step", np.diff(self["surv_bound"])/(self["n_states"]-1),
                "Difference in survival probability between adjacent \
                        reproduction genotype values. [float]")
        self.put("repr_step", np.diff(self["repr_bound"])/(self["n_states"]-1),
                "Difference in reproduction probability between adjacent \
                        reproduction genotype values. [float]")
        self.put("s_range", np.linspace(self["surv_bound"][0],
            self["surv_bound"][1], self["n_states"]), "Survival probability \
                    for each genotype sum value, linearly spaced. \
                    [float array]")
        self.put("r_range", np.linspace(self["repr_bound"][0],
            self["repr_bound"][1], self["n_states"]), "Reproduction \
                    probability for each genotype sum value, linearly spaced. \
                    [float array]")
        # Snapshot stages
        self.put("snapshot_stages", np.around(
            np.linspace(0,self["number_of_stages"]-1,
                self["number_of_snapshots"]), 0).astype(int), "Stages of a \
                        run at which to record detailed information about \
                        the state of the population. [int array]")
        # Params dict
        self.put("params", self.subdict(
            ["repr_mode", "chr_len", "n_base", "maturity", "start_pop",
                "max_ls", "g_dist", "repr_offset", "neut_offset"]),
            "Key information for generating a new population object: \
                    reproduction mode, chromosome length, bases per \
                    chromosome, age of maturity, maximum lifespan, \
                    starting population size, and initial bit value \
                    distribution. [dict]")

        #! Script for generating new config file with default values,
        # or data file with default setup in place
