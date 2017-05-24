########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Config                                                      #
# Description: Simulation configuration information imported from a    #
#   config file and saved as a Python object for use by other classes. #
########################################################################

## PACKAGE IMPORT ##
import numpy as np

## CLASS ##

class Config:
    """Object derived from imported config module."""

    def __init__(self, c):
        """Import basic attributes from a config-file object and set data
        descriptors."""
        # First check that maturity, max_ls anh offset values are compatible
        def mirror(name, infostring):
            """Set a data element's value from the attribute of the same name
            in c, while also setting its description."""
            self.put(name, getattr(c,name), infostring)
        self.info_dict = {} # Dictionary for element information
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
        self.check()

    # Check for invalid config construction
    def check(self):
        """Confirm that population parameters are compatible before
        starting simulation."""
        states = ['asexual','recombine_only','assort_only','sexual']
        if self.repr_mode not in states:
            s0 = "Invalid reproductive mode: '{}'.".format(self.repr_mode)
            s1 = " Must be one of: {}.".format(", ".join(states))
            raise ValueError(s0 + s1)
        if self.maturity > self.max_ls - 2:
            s = "Age of maturity must be at least 2 less than max lifespan."
            raise ValueError(s)
        if self.repr_offset < self.max_ls:
            s = "Offset for reproductive loci must be >= max lifespan."
            raise ValueError(s)
        if self.neut_offset - self.repr_offset < self.max_ls:
            s = "Difference between offset values must be >= max lifespan."
            raise ValueError(s)
        return True

    # Constructors and selectors

    def put(self, name, value, infostring):
        """Set a data element's value and description simultaneously."""
        setattr(self, name, value)
        self.info_dict[name] = infostring

    def get_value(self, key):
        """Get the value of an attribute (equivalent to calling that attribute
        directly."""
        return getattr(self, key)

    def get_info(self, key):
        """Get the description accompanying an attribute."""
        return self.info_dict[key]

    # Generate derived attributes

    def generate(self):
        """Generate derived configuration attributes from simple ones and
        add to configuration object."""
        # Compute values
        self.check()
        # Genome structure
        g_dist_dict = { # Dictionary of initial proportion of 1's in genome loci
                "s":self.g_dist_s, # Survival
                "r":self.g_dist_r, # Reproduction
                "n":self.g_dist_n # Neutral
                }
        self.put("g_dist", g_dist_dict, "Dictionary of values specifying\
                initial proportion of 1's in each locus type (s/r/n) during\
                population initialisation. [dict of floats]")
        genmap = np.concatenate([
            np.arange(self.max_ls),
            np.arange(self.maturity,self.max_ls) + self.repr_offset,
            np.arange(self.n_neutral) + self.neut_offset], 0)
        self.put("genmap", genmap, "Genome map specifying the relative \
                position of each survival, reproduction and neutral locus \
                in the genome of each individual. [int array]")
        self.put("genmap_argsort", np.argsort(self.genmap), "Array specifying\
                how to sort the genome map (or the genome) to correctly order\
                the loci (survival 0-maxls, reproduction maturity-maxls, \
                neutral. [int array]")
        self.put("chr_len", len(self.genmap) * self.n_base, "Length of one\
                chromosome in bits. [int]")
        self.put("n_states", 2*self.n_base+1, "Number of possible genotype \
                states (from 0 to 2*n_base). [int]")
        # Survival and reproduction
        self.death_bound = np.array(self.death_bound)
        self.repr_bound = np.array(self.repr_bound)
        if self.repr_mode in ["sexual", "assort_only"]: 
            # Double fertility in sexual case to control for relative
            # contribution to offspring
            self.repr_bound *= 2 
        self.put("surv_bound", 1-self.death_bound[::-1], "Min and max \
                survival rates. [float array].")
        self.put("surv_step", np.diff(self.surv_bound)/(self.n_states-1),
                "Difference in survival probability between adjacent \
                        reproduction genotype values. [float]")
        self.put("repr_step", np.diff(self.repr_bound)/(self.n_states-1),
                "Difference in reproduction probability between adjacent \
                        reproduction genotype values. [float]")
        self.put("s_range", np.linspace(self.surv_bound[0],self.surv_bound[1],
            self.n_states), "Survival probability for each genotype sum \
                    value, linearly spaced. [float array]")
        self.put("r_range", np.linspace(self.repr_bound[0],self.repr_bound[1],
            self.n_states), "Reproduction probability for each genotype sum \
                    value, linearly spaced. [float array]")
        # Snapshot stages
        self.put("snapshot_stages", np.around(
            np.linspace(0,self.number_of_stages-1,self.number_of_snapshots),
            0).astype(int), "Stages of a run at which to record detailed \
                    information about the state of the population. \
                    [int array]")
        # Params dict
        self.put("params", {
            "repr_mode":self.repr_mode, "chr_len":self.chr_len,
            "n_base":self.n_base, "maturity":self.maturity,
            "max_ls":self.max_ls, "start_pop":self.start_pop,
            "g_dist":self.g_dist, "repr_offset":self.repr_offset,
            "neut_offset":self.neut_offset
            }, "Key information for generating a new population object: \
                    reproduction mode, chromosome length, bases per \
                    chromosome, age of maturity, maximum lifespan, \
                    starting population size, and initial bit value \
                    distribution. [dict]")

        #! Script for generating new config file with default values,
        # or data file with default setup in place
