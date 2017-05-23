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
        def mirror(self, name, infostring):
            """Set a data element's value from the attribute of the same name
            in c, while also setting its description."""
            self.put(name, getattr(c,name), infostring)
        self.info_dir = {} # Dictionary for element information
        # Set data elements
        self.mirror("repr_mode", "The reproductive mode of the population:\
                'sexual' (recombination and assortment), 'recombine_only',\
                'assort_only', or 'asexual' (no recombination or assortment).\
                [string]")
        self.mirror("number_of_runs", "The number of runs to be 
            executed in parallel to generate summary data. [int]")
        self.mirror("number_of_stages", "The number of stages per run. [int]")
        self.mirror("number_of_snapshots", "The number of stages in the run\
                at which to save detailed data (evenly distributed). [int]")
        self.mirror("res_start", "The resource level at stage 0. [int]")
        self.mirror("res_var", "Whether resources vary through consumption\
                and regrowth (True) or remain constant (False). [bool]")
        self.mirror("res_limit", "Maximum resource level (-1 = infinite).\
                [int]") #!
        self.mirror("R", "Arithmetic regrowth increment under variable\
                resources. [int]")
        self.mirror("V", "Geometric regrowth rate under variable\
                resources. [float]")
        self.mirror("start_pop", "Population size at stage 0. [int]")
        self.mirror("g_dist_s", "Initial proportion of 1's in survival loci\
                at stage 0. [float]")
        self.mirror("g_dist_r", "Initial proportion of 1's in reproduction\
                loci at stage 0. [float]")
        self.mirror("g_dist_n", "Initial proportion of 1's in neutral loci\
                at stage 0. [float]")
        self.mirror("death_bound", "Min and max death rates. [float array]")
        self.mirror("repr_bound", "Min and max reproduction rates.\
            [float array]")
        self.mirror("r_rate", "Per-bit recombination rate, if applicable. \
                [float]")
        self.mirror("m_rate", "Per-bit mutation rate during reproduction. \
                [float]")
        self.mirror("m_ratio", "The ratio of positive (0 -> 1) to negative\
                (1 -> 0) mutations during reproduction. Typically negative.\
                [float]")
        self.mirror("max_ls", "Maximum lifespan for the population in stages\
                (-1 = indefinite). [int]") #!
        self.mirror("maturity", "Age of reproductive maturation in stages;\
                before this, P(reproduction) = 0. [int]")
        self.mirror("n_neutral", "Number of neutral loci per genome;\
                more loci reduces the effect of linkage at any given one.\
                [int]")
        self.mirror("n_base", "Number of bits per locus in the genome. [int]")
        self.mirror("surv_pen", "Whether to penalise survival under \
                starvation by multiplying the death rate. [bool]")
        self.mirror("repr_pen", "Whether to penalise reproduction under \
                starvation by dividing the reproduction rate. [bool]")
        self.mirror("death_inc", "Factor for multiplying death rate of \
                individuals under starvation. [float]")
        self.mirror("repr_dec", "Factor for dividing reproduction rate of \
                individuals under starvation. [float]")
        self.mirror("windows", "Width of sliding windows for recording \
                along-genome variation in bit value and along-run \
                variation in population size and resource levels.")

    # Constructors and selectors

    def put(self, name, value, infostring):
        """Set a data element's value and description simultaneously."""
        setattr(self, name, value)
        self.info_dir[name] = infostring

    def get_value(self, key):
        """Get the value of an attribute (equivalent to calling that attribute
        directly."""
        return getattr(self, key)

    def get_info(self, key):
        """Get the description accompanying an attribute."""
        return self.info_dir[key]

    # Generate derived attributes

    def generate(self):
        """Generate derived configuration attributes from simple ones and
        add to configuration object."""

        self.g_dist = { # Dictionary of initial proportion of 1's in genome loci
                "s":self.g_dist_s, # Survival
                "r":self.g_dist_r, # Reproduction
                "n":self.g_dist_n # Neutral
                }
        # Genome map: survival (0 to max), reproduction (maturity to max), neutral
        self.genmap = np.asarray(range(0,self.max_ls) +\
                range(self.maturity+100,self.max_ls+100) +\
                range(200, 200+self.n_neutral))
        # Map from genmap to ordered loci:
        self.genmap_argsort = np.argsort(self.genmap)
        if self.sexual: self.repr_bound *= 2 # x2 fertility rate in sexual case
        # Length of chromosome in binary units
        self.chr_len = len(self.genmap) * self.n_base 
        # Probability ranges for survival and death (linearly-spaced between limits)
        self.n_states = 2*self.n_base+1
        self.surv_bound = 1-self.death_bound[::-1] # Min/max survival probs
        self.repr_step = np.diff(self.repr_bound)/self.n_states
        self.surv_step = np.diff(self.surv_bound)/self.n_states
        self.s_range = np.linspace(self.surv_bound[0], 
                self.surv_bound[1], self.n_states) # min to max surv rate
        self.r_range = np.linspace(self.repr_bound[0],
                self.repr_bound[1], self.n_states) # min to max repr rate
        # Determine snapshot stages (evenly spaced within run):
        if type(self.number_of_snapshots) is float:
            self.snapshot_proportion = self.number_of_snapshots
            self.number_of_snapshots = int(\
                    self.number_of_snapshots*self.number_of_stages)
        self.snapshot_stages = np.around(np.linspace(0,
            self.number_of_stages-1,self.number_of_snapshots),0).astype(int)
        # Dictionary for population initialisation
        self.params = {"sexual":self.sexual, 
                "chr_len":self.chr_len, "n_base":self.n_base,
                "maturity":self.maturity, "max_ls":self.max_ls,
                "age_random":self.age_random, 
                "start_pop":self.start_pop, "g_dist":self.g_dist}
