# cython: profile=True

# Modules
import numpy as np
cimport numpy as np
import scipy.stats as st
import importlib, operator, time, os, random, datetime, copy, multiprocessing
try:
       import cPickle as pickle
except:
       import pickle

# Cython data types for numpy
NPINT = np.int # Integer arrays
NPFLOAT = np.float # Decimal arrays
NPBOOL = np.uint8 # Boolean arrays

ctypedef np.int_t NPINT_t
ctypedef np.float_t NPFLOAT_t
ctypedef np.uint8_t NPBOOL_t

#############
# FUNCTIONS #
#############

rand = st.uniform(0,1) # Uniform random number generator
def chance(p,n=1):
    """Generate array (of shape specified by n, where n is either an integer
    or a tuple of integers) of independent booleans with P(True)=z."""
    return rand.rvs(n)<p

def get_runtime(starttime, endtime):
    """Convert two datetime.now() outputs into a time difference 
    in human-readable units."""
    runtime = endtime - starttime
    days = runtime.days
    hours = runtime.seconds/3600
    minutes = runtime.seconds/60 - hours*60
    seconds = runtime.seconds - minutes*60 - hours*3600
    delta = "Total runtime: "
    if days != 0: delta += "{d} days, ".format(d=days)
    if hours != 0: delta += "{h} hours, ".format(h=hours)
    if minutes != 0: delta += "{m} minutes, ".format(m=minutes)
    delta += "{s} seconds.".format(s=seconds)
    return delta

def execute_run(run, maxfail):
    """Execute a simulation run, handling and recording failed runs
    as appropriate."""
    if maxfail>0: blank_run = copy.deepcopy(run)
    run.execute()
    if maxfail>0 and run.dieoff:
        nfail = blank_run.record.record["prev_failed"]
        if nfail >= maxfail - 1:
            run.logprint("Total failures = {0}.".format(nfail+1))
            run.logprint("Failure limit reached. Accepting failed run.")
            run.logprint(get_runtime(run.starttime, run.endtime))
            return run
        nfail += 1
        run.logprint("Run failed. Total failures = {0}. Repeating..."\
                .format(nfail))
        blank_run.record.record["prev_failed"] = nfail
        def divplus1(x): return (x*100)/(x+1)
        blank_run.record.record["percent_dieoff"] = \
                divplus1(blank_run.record.record["prev_failed"])
        blank_run.log = run.log + "\n"
        blank_run.starttime = run.starttime
        return execute_run(blank_run, maxfail)
    run.logprint(get_runtime(run.starttime, run.endtime))
    return run

def testage():
    return np.array([-1]) # For generating new populations
def testgen():
    return np.array([[-1],[-1]])

###############################################################################
# CLASSES 
###############################################################################
#
#  Currently Simulation, Run, Config, Outpop, Population & Record
#
#  Simulation: Initialises and finalises the simulation; wrapper for one or
#    more Run objects.
#  Run: Executes simulated evolution of a population from start to finish and
#    records the results
#  Config: Contains configuration information from a config file, plus derived
#    parameters.
#  Outpop: Stores the current state of a simulated population during
#    initialisation, cleanup and storage; contains information about the state
#    of a population but no update methods (growth, death, etc.)
#  Population: Stores the current state of a simulated population during the
#    execution of a Run and updates it (growth, death, etc.) over the course of
#    the simulation.
#  Record: Stores information about the evolution of a Population over the 
#    course of a Run.
#
#  Simulation ---> Config
#              |-> Outpop (Starting population, inherited by Runs)
#              |-> Run --> Config (inherited from Simulation)
#                      |-> Outpop (Starting and final population)
#                            |--> Population (for execution only)
#                      |-> Record
#
###############################################################################

class Config:
    """Object derived from imported config module."""
    def __init__(self, c):
        self.sexual = c.sexual # Sexual or asexual reproduction
        self.number_of_runs = c.number_of_runs # Total number of runs
        self.number_of_stages = c.number_of_stages # Number of stages per run
        self.crisis_p = c.crisis_p # Per-stage crisis probability
        self.crisis_stages = c.crisis_stages # Stages with guaranteed crises
        self.crisis_sv = c.crisis_sv # Proportion of crisis survivors
        self.number_of_snapshots = c.number_of_snapshots # Total snapshots per run
        self.res_start = c.res_start # Initial resource value
        self.res_var = c.res_var # Whether resources vary with time
        self.res_limit = c.res_limit # Maximum resource value, if variable
        self.R = c.R # Arithmetic resource increment factor, if variable
        self.V = c.V # Geometric resource increment factor, if variable
        self.start_pop = c.start_pop # Starting population size
        self.age_random = c.age_random # Whether new population has random ages
        self.g_dist_s = c.g_dist_s # Proportion of 1's in initial survival loci
        self.g_dist_r = c.g_dist_r # --""-- reproductive loci
        self.g_dist_n = c.g_dist_n # --""-- neutral loci
        self.death_bound = c.death_bound # Min and max death rates
        self.repr_bound = c.repr_bound # Min and max reproduction rates
        self.r_rate = c.r_rate # Per-bit recombination rate, if sexual
        self.m_rate = c.m_rate # Per-bit mutation rate during reproduction
        self.m_ratio = c.m_ratio # Positive:negative mutation ratio
        self.max_ls = c.max_ls # Maximum lifespan; must be less than 100
        self.maturity = c.maturity # Age of sexual maturation
        self.n_neutral = c.n_neutral # Number of neutral loci
        self.n_base = c.n_base # Bits per locus
        self.surv_pen = c.surv_pen # Penalise survival under starvation?
        self.repr_pen = c.repr_pen # Penalise reproduction under starvation?
        self.death_inc = c.death_inc # Rate of death-rate increase under starvation
        self.repr_dec = c.repr_dec # Rate of repr rate decrease under starvation
        self.window_size = c.window_size # Sliding window size for SD recording

    def generate(self):
        """Generate derived configuration parameters from simple ones and
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
        if self.sexual: self.repr_bound[1] *= 2 # x2 fertility rate in sexual case
        # Length of chromosome in binary units
        self.chr_len = len(self.genmap) * self.n_base 
        # Probability ranges for survival and death (linearly-spaced between limits)
        nstates = 2*self.n_base+1
        self.d_range = np.linspace(self.death_bound[1], 
                self.death_bound[0], nstates) # max to min death rate
        self.r_range = np.linspace(self.repr_bound[0],
                self.repr_bound[1], nstates) # min to max repr rate
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

class Outpop:
    """Pickle-able I/O form of Population class."""
    def __init__(self, pop):
        """Generate an Outpop from a Population object."""
        self.sex = pop.sex
        self.nbase = pop.nbase
        self.chrlen = pop.chrlen
        self.maxls = pop.maxls
        self.maturity = pop.maturity
        self.genmap = np.copy(pop.genmap)
        self.ages = np.copy(pop.ages)
        self.genomes = np.copy(pop.genomes)
        self.N = pop.N

    def params(self):
        """Report fixed population-intrinsic parameters."""
        p_dict = {
                "sexual":self.sex,
                "chr_len":self.chrlen,
                "n_base":self.nbase,
                "max_ls":self.maxls,
                "maturity":self.maturity
                }
        return p_dict

    def toPop(self):
        """Make cythonised Population object from this Outpop."""
        return Population(self.params(), self.genmap, self.ages, 
                self.genomes)

    def clone(self):
        """Generate a new, identical Outpop object."""
        return Outpop(self)

cdef class Population:
    """A simulated population with genomes and ages."""
    cdef public np.ndarray genmap, ages, genomes, genmap_argsort
    cdef public int sex, chrlen, nbase, maxls, maturity, N
    # Initialisation
    def __init__(self, dict params, np.ndarray[NPINT_t, ndim=1] genmap, 
            np.ndarray[NPINT_t, ndim=1] ages, 
            np.ndarray[NPINT_t, ndim=2] genomes):
        """Create a new population, either with newly-generated age and genome
        vectors or inheriting these from a seed."""
        self.sex = params["sexual"] * 1
        self.nbase = params["n_base"]
        self.chrlen = params["chr_len"]
        self.maxls = params["max_ls"]
        self.maturity = params["maturity"]
        self.genmap = genmap
        self.genmap_argsort = np.argsort(genmap)
        # Determine ages and genomes if not given
        if params.has_key("start_pop"):
            if np.array_equal(ages, testage()):
                ages = np.random.random_integers(0, self.maxls-1,
                        params["start_pop"]) if params["age_random"]\
                        else np.repeat(self.maturity,params["start_pop"])
            if np.array_equal(genomes, testgen()):
                genomes = self.make_genome_array(
                        params["start_pop"], params["g_dist"])
        self.ages = np.copy(ages)
        self.genomes = np.copy(genomes)
        self.N = len(self.ages)

    # Minor methods

    def shuffle(self):
        """Rearrange order of individuals in population."""
        index = np.arange(self.N)
        np.random.shuffle(index)
        self.ages = self.ages[index]
        self.genomes = self.genomes[index]

    def clone(self):
        """Generate a new, identical population object."""
        return Population(self.params(), self.genmap,
                self.ages, self.genomes)

    def increment_ages(self):
        """Age all individuals in population by one stage."""
        self.ages += 1

    def params(self):
        """Get population-intrinsic parameters from Population object."""
        p_dict = {
                "sexual":self.sex,
                "chr_len":self.chrlen,
                "n_base":self.nbase,
                "max_ls":self.maxls,
                "maturity":self.maturity
                }
        return p_dict

    cpdef addto(self, object pop):
        """Append the individuals from a second population to this one,
        keeping this one's parameters and genome map."""
        self.ages = np.concatenate((self.ages, pop.ages), 0)
        self.genomes = np.concatenate((self.genomes, pop.genomes), 0)
        self.N = len(self.ages)
        # As it stands cythonising this doesn't make much difference

    def chrs(self, reshape=False):
        """Return an array containing the first and second chromosomes 
        of each individual in the population, in either 2D individual-x-bit
        or 3D individual-x-locus-x-bit configuration."""
        if not reshape:
            # dim1=chromosome, dim2=individual, dim3=bit
            return self.genomes.reshape((self.N,2,self.chrlen)
                    ).transpose((1,0,2))
        else:
            # dim1=chromosome, dim2=individual, dim3=locus, dim4=bit
            return self.genomes.reshape((self.N,2,len(self.genmap),self.nbase)
                    ).transpose(1,0,2,3)
        # Not happy with the transposition efficiency-wise, but having 
        # individuals first screws up recombination/assortment in ways I
        # don't know how to efficiently fix...

    def sorted_loci(self):
        """Return the sorted locus genotypes of the individuals in the 
        population, summed within each locus and across chromosomes."""
        # Get chromosomes of population, arranged by locus
        chrx = self.chrs(True) 
        # Collapse bits into locus sums and add chromosome values together
        # to get total genotype value for each locus (dim1=indiv, dim2=locus)
        locs = np.einsum("ijkl->jk", chrx)
        return locs[:,self.genmap_argsort]

    # Major methods:

    def make_genome_array(self, start_pop, g_dist):
        """
        Generate genomes for start_pop individuals with chromosome length of 
        chr_len bits and locus length of n_base bits. Set the genome array 
        values (distribution of 1's and 0's) according to genmap and g_dist.
        """
        # Initialise genome array:
        genome_array = np.zeros([start_pop, self.chrlen*2])
        # Use genome map to determine probability distribution for each locus:
        loci = {
            "s":np.nonzero(self.genmap<100)[0],
            "r":np.nonzero(np.logical_and(self.genmap>=100,self.genmap<200))[0],
            "n":np.nonzero(self.genmap>=200)[0]
            }
        # Set genome array values according to given probabilities:
        for k in loci.keys():
            # Identify genome positions corresponding to locus type
            pos = np.array([range(self.nbase) + x for x in loci[k]*self.nbase])
            pos = np.append(pos, pos + self.chrlen)
            # Add values to positions according to appropriate distribution
            genome_array[:, pos] = chance(g_dist[k], [start_pop, len(pos)])
        return genome_array.astype("int")

    cpdef get_subpop(self, int min_age, int max_age, int offset,
            np.ndarray[NPFLOAT_t, ndim=1] val_range):
        """Select a population subset based on chance defined by genotype."""
        cdef:
            int age, locus, g
            np.ndarray[NPINT_t, ndim=1] which
            np.ndarray[NPFLOAT_t, ndim=1] inc_probs
            np.ndarray[NPINT_t, ndim=3] genloc, pop
            np.ndarray[NPBOOL_t, ndim=1,cast=True] inc
        g = len(self.genmap)
        # Probability of each individual being included in subpop (default 0)
        inc_probs = np.zeros(self.N)
        # Reshape genome array such that dim1=individuals, dim2=loci, dim3=bits
        genloc = np.reshape(self.genomes, (self.N, g*2, self.nbase))
        # Get inclusion probabilities age-wise:
        for age in range(min_age, min(max_age, np.max(self.ages)+1)):
            # Get indices of appropriate locus for that age:
            locus = np.ndarray.nonzero(self.genmap==(age+offset))[0][0]
            # NB: Will only return FIRST locus for that age in each chromosome
            # Subset to genome positions in required locus, on both chromosomes,
            # for individuals of correct age
            which = np.nonzero(self.ages == age)[0]
            pop = genloc[which][:,[locus, locus+g]]
            # Determine inclusion probabilities for these individuals based on
            # their locus genotype and the value range given
            inc_probs[which] = val_range[np.einsum("ijk->i", pop)]
        inc = chance(inc_probs, self.N)
        return inc # Binary array giving inclusion status of each individuals

    cpdef growth(self, np.ndarray[NPFLOAT_t, ndim=1] var_range, float penf, 
            float r_rate, float m_rate, float m_ratio):
        """Generate new mutated children from selected parents."""
        cdef:
            np.ndarray[NPBOOL_t, ndim=1,cast=True] which_parents
            object parents, children
        if self.N == 0: return # Insulate from empty population errors
        r_range = np.clip(var_range / penf, 0, 1) # Limit to real probabilities
        which_parents = self.get_subpop(self.maturity, self.maxls, 100, 
                r_range) # Get subpopulation status of parents
        if self.sex:
            if sum(which_parents) == 1:
                return # No children if only one parent
            else: # Generate parent population, pair, recombine and assort
                parents = Population(self.params(), self.genmap,
                        self.ages[which_parents],self.genomes[which_parents])
                parents.recombine(r_rate)
                children = parents.assortment()
        else: # Copy parents asexually
            children = Population(self.params(), self.genmap,
                    self.ages[which_parents], self.genomes[which_parents])
        children.mutate(m_rate, m_ratio) # Mutate children
        children.ages[:] = 0 # Make newborn
        self.addto(children) # Add to population

    cpdef death(self, np.ndarray[NPFLOAT_t, ndim=1] var_range, float penf):
        """Select survivors and kill rest of population."""
        cdef:
            np.ndarray[NPFLOAT_t, ndim=1] d_range
            np.ndarray[NPBOOL_t, ndim=1,cast=True] survivors
            int new_N, dead
        if self.N == 0: return # Insulate from empty population errors
        d_range = np.clip(var_range*penf, 0, 1) # Limit to real probabilities
        # Generate survivor array using inverted death probabilities:
        survivors = self.get_subpop(0, self.maxls, 0, 1-d_range)
        # Subset population to survivors:
        self.ages = self.ages[survivors]
        self.genomes = self.genomes[survivors]
        self.N = np.sum(survivors)

    def crisis(self, crisis_sv):
        """Apply an extrinsic death crisis and subset population."""
        #! Currently all individuals have equal chance to survive; change this?
        if self.N == 0: return # Insulate from empty population errors
        n_survivors = int(self.N*crisis_sv)
        which_survive = np.random.choice(np.arange(self.N), n_survivors, False)
        self.ages = self.ages[which_survive]
        self.genomes = self.genomes[which_survive]
        self.N = len(self.ages)

    cpdef recombine(self, float r_rate):
        """Recombine between the two chromosomes of each individual
        in the population."""
        cdef:
            np.ndarray[NPINT_t, ndim=2] r_sites, which_chr
            np.ndarray[NPINT_t, ndim=3] chrs
        if (r_rate > 0): # Otherwise do nothing
            # Randomly generate recombination sites
            r_sites = chance(r_rate, [self.N, self.chrlen]).astype(int)
            # Convert into [1,-1]
            r_sites = np.array([1,-1])[r_sites]
            # Determine crossover status of each position (1 = no crossover,
            # -1 = crossover)
            which_chr = np.cumprod(r_sites, 1)
            # Convert to 0 = no crossover, 1 = crossover
            which_chr = 1 - (which_chr+1)/2
            # Generate new chromosomes and update genomes
            chrs = np.copy(self.chrs())
            self.genomes[:,:self.chrlen] = np.choose(which_chr, chrs)
            self.genomes[:,self.chrlen:] = np.choose(which_chr, chrs[[1,0]])

    def assortment(self):
        """Pair individuals into breeding pairs and generate children
        through random assortment."""
        pop = self.clone()
        # Must be even number of parents; if odd, discard one at random:
        if pop.N%2 != 0:
            ix = random.sample(range(pop.N), 1)
            pop.genomes = np.delete(pop.genomes, ix, 0)
            pop.N -= 1
        # Randomly assign mating partners:
        pop.shuffle()
        # Randomly combine parental chromatids
        chrs = np.copy(pop.chrs())
        which_pair = np.arange(pop.N/2)*2 # First pair member (0,2,4,...)
        which_partner = chance(0.5,pop.N/2) # Member within pair (0 or 1)
        # Update population
        pop.genomes[::2,:self.chrlen] = chrs[0,which_pair+which_partner]
        pop.genomes[::2,self.chrlen:] = chrs[1,which_pair+(1-which_partner)]
        pop.genomes = pop.genomes[::2]
        pop.ages = pop.ages[::2] # Doesn't really matter, will be zero'd
        pop.N /= 2
        return(pop)

    cpdef mutate(self, float m_rate, float m_ratio):
        """Mutate genomes of population according to stated rates."""
        cdef:
            np.ndarray[NPBOOL_t, ndim=2,cast=True] is_0, is_1
            np.ndarray[NPINT_t, ndim=1] positive_mut, negative_mut
        if m_rate > 0: # Else do nothing
            is_0 = self.genomes==0
            is_1 = np.invert(is_0)
            positive_mut = chance(m_rate*m_ratio, np.sum(is_0)).astype(int)
            negative_mut = 1-chance(m_rate, np.sum(is_1))
            self.genomes[is_0] = positive_mut
            self.genomes[is_1] = negative_mut

class Record:
    """An enhanced dictionary object recording simulation data."""
    def __init__(self, conf):
        """Initialise record object and specify initial values."""
        self.record = {}
        # Basic run info
        self.record["dieoff"] = False
        self.record["prev_failed"] = 0 # DOES NOT include current run if it fails
        self.record["percent_dieoff"] = 0
        # Population parameters from config object
        self.record["genmap"] = conf.genmap
        self.record["genmap_argsort"] = np.argsort(conf.genmap)
        self.record["chr_len"] = np.array([conf.chr_len])
        self.record["n_bases"] = np.array([conf.n_base])
        self.record["max_ls"] = np.array([conf.max_ls])
        self.record["maturity"] = np.array([conf.maturity])
        self.record["sexual"] = conf.sexual
        # Run parameters from config object
        self.record["d_range"] = conf.d_range
        self.record["r_range"] = conf.r_range
        self.record["snapshot_stages"] = conf.snapshot_stages
        self.record["n_snapshots"] = conf.number_of_snapshots
        self.record["n_stages"] = conf.number_of_stages
        self.record["res_var"] = conf.res_var
        # Data collected at every stage
        per_stage_array = np.zeros(conf.number_of_stages)
        self.record["population_size"] = np.copy(per_stage_array)
        self.record["resources"] = np.copy(per_stage_array)
        self.record["surv_penf"] = np.copy(per_stage_array)
        self.record["repr_penf"] = np.copy(per_stage_array)
        self.record["age_distribution"] = np.zeros([conf.number_of_stages,
            conf.max_ls])
        # Data collected for each age at each snapshot
        m = len(conf.snapshot_stages)
        age_snapshot_array = np.zeros([m,conf.max_ls])
        keys = ["death_mean","death_sd","repr_mean","repr_sd",
                "age_wise_fitness_product","age_wise_fitness_contribution",
                "junk_age_wise_fitness_product",
                "junk_age_wise_fitness_contribution"]
        for k in keys: self.record[k] = np.copy(age_snapshot_array)
        # Genotype distribution data collected at each snapshot
        gt_snapshot_array = np.zeros([m,2*conf.n_base+1])
        ## Distribution of locus sums, from min to max total value
        self.record["density_surv"] = np.copy(gt_snapshot_array)
        self.record["density_repr"] = np.copy(gt_snapshot_array)
        ## Number of 1's at each position along the chromosome
        self.record["n1"] = np.zeros([m,conf.chr_len])
        ## ?
        self.record["n1_std"] = np.zeros([m,conf.chr_len])
        ## ?
        self.record["s1"] = np.zeros([m,conf.chr_len-conf.window_size+1])
        # Simple per-snapshot data
        snapshot_array = np.zeros(m)
        keys = ["entropy","junk_death","junk_repr","junk_fitness","fitness",
                "junk_death_sd", "junk_repr_sd"]
        for k in keys: self.record[k] = np.copy(snapshot_array)

    def quick_update(self, n_stage, population, resources, surv_penf, repr_penf):
        """Record only per-stage data, i.e. population size, age distribution, 
        resources, and starvation penalties."""
        self.record["population_size"][n_stage] = population.N
        self.record["resources"][n_stage] = resources
        self.record["surv_penf"][n_stage] = surv_penf
        self.record["repr_penf"][n_stage] = repr_penf
        self.record["age_distribution"][n_stage] = np.bincount(population.ages,
                minlength = population.maxls)/float(population.N)

    def full_update(self, population, n_snap):
        """Record detailed per-age statistics of population at
        current snapshot stage, inc. death rate, reproduction rate, genotype
        density."""
        p = population
        b,g = p.nbase,len(p.genmap) # Number of bits per locus
        locs = p.sorted_loci()
        # Subset to survival/reproductive/neutral loci
        surv_locs = locs[:,:p.maxls]
        repr_locs = locs[:,p.maxls:(2*p.maxls-p.maturity)]
        neut_locs = locs[:,(2*p.maxls-p.maturity):]
        #repr_locs = locs[:,np.arange(p.maturity,p.maxls)+p.maxls]
        # Calculate overall genotype distribution of surv/repr loci
        def density(loci):
            """Return the normalised distributions of sum genotypes of an array
            of genomic loci."""
            bins = np.bincount(np.ndarray.flatten(locs),minlength=2*b+1)
            return bins/float(sum(bins))
        self.record["density_surv"][n_snap] += density(surv_locs)
        self.record["density_repr"][n_snap] += density(repr_locs)
        self.record["entropy"][n_snap] = st.entropy(density(locs))
        # Convert genotypes to selection rates and find per-age mean and SD
        death_rates = self.record["d_range"][surv_locs]
        repr_rates = np.zeros([p.N, p.maxls])
        repr_rates[:,p.maturity:] += self.record["r_range"][repr_locs]
        self.record["death_mean"][n_snap] += np.mean(death_rates, 0)
        self.record["death_sd"][n_snap] += np.std(death_rates, 0)
        self.record["repr_mean"][n_snap] += np.mean(repr_rates, 0)
        self.record["repr_sd"][n_snap] += np.std(repr_rates, 0)
        # Junk stats calculated from neutral locus
        death_rates_neut = self.record["d_range"][neut_locs]
        repr_rates_neut = self.record["r_range"][neut_locs]
        self.record["junk_death"][n_snap] = np.mean(death_rates_neut)
        self.record["junk_death_sd"][n_snap] = np.std(death_rates_neut)
        self.record["junk_repr"][n_snap] = np.mean(repr_rates_neut)
        self.record["junk_repr_sd"][n_snap] = np.std(repr_rates_neut)
        # Calculate mean and SD of number of 1's at each chromosome position,
        # ordered according to genome map
        chr_pos = p.genomes.reshape(p.N*2,p.chrlen)
        # Mean and SD number of 1's per chromosome bit, ordered by genome map
        order = np.ndarray.flatten(
            np.array([p.genmap_argsort*b + c for c in xrange(b)]),
            order="F")
        self.record["n1"][n_snap] = np.mean(chr_pos, axis=0)[order]
        self.record["n1_std"][n_snap] = np.std(chr_pos, axis=0)[order]

    def update(self, population, resources, surv_penf, repr_penf, stage, n_snap,
           full_update):
        """Record detailed population data at current snapshot stage."""
        self.quick_update(stage, population, resources, surv_penf, repr_penf)
        if full_update: self.full_update(population, n_snap)

    def age_wise_n1(self, arr_str):
        """Convert n1/n1_std array from value-per-bit to value-per-age."""
        arr = self.record[arr_str] # already sorted
        s, b = arr.shape, self.record["n_bases"]
        res = np.mean(arr.reshape((s[0], self.record["chr_len"]/b, b)), 2)
        return res

    def actual_death_rate(self):
        """Compute actual death rate for each age at each stage."""
        N_age = self.record["age_distribution"] *\
                self.record["population_size"][:,None]
        dividend = N_age[1:, 1:]
        divisor = np.copy(N_age[:-1, :-1])
        divisor[divisor == 0] = 1 # avoid division by zero
        death = 1 - dividend / divisor
        # value for last age is 1
        return np.append(death, np.ones([death.shape[0], 1]), axis=1)

    def final_update(self, window):
        """Compute end-of-run statistics: fitness, rolling SD of n1, 
        actual survival, etc."""
        r = self.record
        # Rolling standard deviation of #1's along genome:
        d,s = r["n1"].ndim-1, len(r["n1"].strides)-1
        a_shape = r["n1"].shape[:d] + (r["n1"].shape[d] - window + 1, window)
        a_strd = r["n1"].strides + (r["n1"].strides[s],) # strides
        r["s1"] = np.std(np.lib.stride_tricks.as_strided(
            r["n1"], shape=a_shape, strides=a_strd), 2)
        # Calculate fitness measures
        norm_surv,norm_repr = 1-r["death_mean"],r["repr_mean"]
        norm_repr[:,:r["maturity"]] = 0 # (repr before maturity = 0)
        r["age_wise_fitness_product"] = norm_surv * norm_repr
        r["age_wise_fitness_contribution"] = np.cumprod(norm_surv,1)*norm_repr
        r["fitness"] = np.sum(r["age_wise_fitness_contribution"],1)
        # Junk fitness
        def jtile(x): return np.tile(x.reshape(x.shape[0],1),r["max_ls"])
        # (tile to shape (n_snap,maxls) - IMPORTANT)
        junk_surv_n,junk_repr_n = jtile(1-r["junk_death"]),jtile(r["junk_repr"])
        junk_repr_n[:,:r["maturity"]] = 0 # (repr before maturity = 0)
        r["junk_age_wise_fitness_product"] = junk_surv_n * junk_repr_n
        r["junk_age_wise_fitness_contribution"] = \
                np.cumprod(junk_surv_n,1)*junk_repr_n
        r["junk_fitness"] = np.sum(r["junk_age_wise_fitness_contribution"],1)
        # Halve fitness values for sexual populations
        if r["sexual"]:
            for k in ["age_wise_fitness_product","age_wise_fitness_contribution",
                    "fitness","junk_age_wise_fitness_product","junk_fitness",
                    "junk_age_wise_fitness_contribution"]:
                r[k] /= 2.0
        # Other stats
        self.record["actual_death_rate"] = self.actual_death_rate()
        self.record["age_wise_n1"] = self.age_wise_n1("n1")
        self.record["age_wise_n1_std"] = self.age_wise_n1("n1_std")

class Run:
    """An object representing a single run of a simulation."""
    def __init__(self, config, startpop, n_run, report_n, verbose):
        self.log = ""
        self.conf = copy.deepcopy(config)
        self.surv_penf = 1.0
        self.repr_penf = 1.0
        self.resources = self.conf.res_start
        np.random.shuffle(self.conf.genmap)
        self.genmap = self.conf.genmap # Not a copy
        if startpop != "":
            self.population = startpop.clone()
            # Adopt from population: genmap, nbase, chrlen
            self.conf.genmap = self.population.genmap
            self.conf.chr_len = self.population.chrlen
            self.conf.n_base = self.population.nbase
            # Keep new max_ls, maturity, sexual
            self.population.maturity = self.conf.maturity
            self.population.maxls = self.conf.max_ls
            self.population.sex = self.conf.sexual
            self.conf.params = {"sexual":self.conf.sexual,
                    "chr_len":self.conf.chr_len, "n_base":self.conf.n_base,
                    "maturity":self.conf.maturity, "max_ls":self.conf.max_ls,
                    "age_random":self.conf.age_random,
                    "start_pop":self.conf.start_pop, "g_dist":self.conf.g_dist}
        else:
            self.population = Outpop(Population(self.conf.params,
                self.conf.genmap, testage(), testgen()))
        self.n_stage = 0
        self.n_snap = 0
        self.n_run = n_run
        self.record = Record(self.conf)
        self.dieoff = False
        self.complete = False
        self.report_n = report_n
        self.verbose = verbose

    def update_resources(self):
        """If resources are variable, update them based on current 
        population."""
        if self.conf.res_var: # Else do nothing
            k = 1 if self.population.N > self.resources else self.conf.V
            new_res = int((self.resources-self.population.N)*k + self.conf.R)
            self.resources = np.clip(new_res, 0, self.conf.res_limit)

    def starving(self):
        """Determine whether population is starving based on resource level."""
        if self.conf.res_var:
            return self.resources == 0
        else:
            return self.population.N > self.resources

    def update_starvation_factors(self):
        """Update starvation factors under starvation."""
        if self.starving():
            self.surv_penf *= self.conf.death_inc if self.conf.surv_pen else 1.0
            self.repr_penf *= self.conf.repr_dec if self.conf.repr_pen else 1.0
        else: 
            self.surv_penf = 1.0
            self.repr_penf = 1.0

    def execute_stage(self):
        """Perform one stage of a simulation run and test for completion."""
        if not isinstance(self.population, Population):
            m="Convert Outpop objects to Population before running execute_stage."
            raise TypeError(m)
        report_stage = (self.n_stage % self.report_n == 0)
        if report_stage:
            self.logprint("Population = {0}.".format(self.population.N))
        self.dieoff = (self.population.N == 0)
        if not self.dieoff:
            # Record information
            take_snapshot = self.n_stage in self.conf.snapshot_stages
            full_report = report_stage and self.verbose
            self.record.update(self.population, self.resources, self.surv_penf,
                    self.repr_penf, self.n_stage, self.n_snap, take_snapshot)
            self.n_snap += 1 if take_snapshot else 0
            if take_snapshot and full_report: self.logprint("Snapshot taken.")
            # Update ages, resources and starvation
            self.population.increment_ages()
            self.update_resources()
            self.update_starvation_factors()
            if full_report: self.logprint(
                    "Starvation factors = {0} (survival), {1} (reproduction)."\
                            .format(self.surv_penf,self.repr_penf))
            # Reproduction and death
            if full_report: self.logprint("Calculating reproduction and death...")
            n0 = self.population.N
            self.population.growth(self.conf.r_range, self.repr_penf,
                    self.conf.r_rate, self.conf.m_rate, self.conf.m_ratio)
            n1 = self.population.N
            self.population.death(self.conf.d_range, self.surv_penf)
            n2 = self.population.N
            if full_report: 
                self.logprint("Done. {0} individuals born, {1} died."\
                        .format(n1-n0,n1-n2))
            if self.n_stage in self.conf.crisis_stages or chance(self.conf.crisis_p):
                self.population.crisis(self.conf.crisis_sv)
                self.logprint("Crisis! {0} individuals died, {1} survived."\
                        .format(n2-self.population.N, self.population.N))
        # Update run status
        self.dieoff = self.record.record["dieoff"] = (self.population.N == 0)
        self.record.record["percent_dieoff"] = self.dieoff*100.0
        self.n_stage += 1
        self.complete = self.dieoff or self.n_stage==self.conf.number_of_stages
        if self.complete and not self.dieoff:
            self.record.final_update(self.conf.window_size)

    def execute(self):
        """Execute a run object from start to completion."""
        self.population = self.population.toPop()
        if not hasattr(self, "starttime"): 
            self.starttime = datetime.datetime.now()
        runstart = time.strftime('%X %x', time.localtime())
        f,r = self.record.record["prev_failed"]+1, self.n_run
        a = "run {0}, attempt {1}".format(r,f) if f>1 else "run {0}".format(r)
        self.logprint("Beginning {0} at {1}.".format(a, runstart))
        while not self.complete:
            self.execute_stage()
        self.population = Outpop(self.population)
        self.endtime = datetime.datetime.now()
        runend = time.strftime('%X %x', time.localtime())
        b = "Extinction" if self.dieoff else "Completion"
        self.logprint("{0} at {1}. Final population: {2}"\
                .format(b, runend, self.population.N))
        if f>0 and not self.dieoff: 
            self.logprint("Total attempts required: {0}.".format(f))

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        # Compute numbers of spaces to keep all messages aligned
        nspace_run = len(str(self.conf.number_of_runs-1))-\
                len(str(self.n_run))
        nspace_stg = len(str(self.conf.number_of_stages-1))\
                -len(str(self.n_stage))
        # Create string
        lstr = "RUN {0}{1} | STAGE {2}{3} | {4}".format(" "*nspace_run, 
                self.n_run, " "*nspace_stg, self.n_stage, message)
        print lstr
        self.log += lstr+"\n"

class Simulation:
    """An object representing a simulation, as defined by a config file."""

    def __init__(self, config_file, seed, seed_n, report_n, verbose):
        self.starttime = datetime.datetime.now()
        simstart = time.strftime('%X %x', time.localtime())+".\n"
        self.log = ""
        self.logprint("\nBeginning simulation at "+simstart)
        self.logprint("Working directory: "+os.getcwd())
        self.get_conf(config_file)
        self.conf.generate()
        self.get_startpop(seed, seed_n)
        self.report_n, self.verbose = report_n, verbose
        self.logprint("Initialising runs...")
        y,x = (len(self.startpop) == 1), xrange(self.conf.number_of_runs)
        self.runs = [Run(self.conf, self.startpop[0 if y else n], n,
            self.report_n, self.verbose) for n in x]
        self.logprint("Runs initialised. Executing...\n")

    def execute(self, nproc=-1, maxfail=10):
        """Execute all runs."""
        if nproc <= 0: # Use all available cores
            pool = multiprocessing.Pool()
        elif nproc == 1: # Run without multiprocessing
            self.runs = [execute_run(r, maxfail) for r in self.runs]
            return
        else: # Use specifed number of cores
            pool = multiprocessing.Pool(nproc)
        lock = multiprocessing.Lock()
        lock.acquire()
        try:
            asyncruns = []
            for n in xrange(self.conf.number_of_runs):
                asyncruns+= [pool.apply_async(execute_run, [self.runs[n],
                    maxfail])]
            outruns = [x.get() for x in asyncruns]
            self.runs = outruns
            self.log += "\n".join([x.log for x in self.runs])
        finally:
            lock.release()

    def get_conf(self, file_name):
        """Import specified configuration file for simulation."""
        try:
            conf = importlib.import_module(file_name)
        except ImportError:
            print "No such file in simulation directory: " + file_name
            q = raw_input(
                    "Enter correct config file name, or skip to abort: ")
            if q == "":
                exit("Aborting: no valid configuration file given.")
            else:
                return self.get_conf(q)
        self.conf = Config(conf)

    def get_startpop(self, seed="", pop_number=-1):
        """Import any seed simulation (or return blank)."""
        if seed == "":
            self.logprint("Seed: None.")
            self.startpop = [""]
            return
        p = "all populations" if pop_number==-1 else "population "+str(pop_number)
        if isinstance(seed, Simulation):
            status = "Seeding {0} directly from Simulation object."
            simobj = seed
        else:
            try:
                # Make sure includes extension (default "sim")
                seed += ".sim" if os.path.splitext(seed)[1] == "" else ""
                status = "Seed: "+seed+", {0}."
                simfile = open(seed, "rb")
                simobj = pickle.load(simfile) # import simulation object
            except IOError:
                print "No such seed file: " + seed
                q = raw_input(
                        "Enter correct path to seed file, or skip to abort: ")
                if q == "": exit("Aborting.")
                r = raw_input("Enter population seed number, or enter -1 to \
                        seed all populations, or skip to abort.")
                if r == "": exit("Aborting.")
                return self.get_startpop(q, r)
        nruns = len(simobj.runs)
        if pop_number == -1:
            # -1 = seed all populations to equivalent runs in new sim
            if nruns != self.conf.number_of_runs:
                message = "Number of runs in seed file ({0}) ".format(nruns)
                message += "does not match current configuration ({0} runs)."\
                        .format(self.conf.number_of_runs)
                raise IndexError(message)
            self.logprint(status.format("all populations"))
            self.startpop = [r.population for r in simobj.runs]
        else:
            if pop_number >= nruns:
                print "Population seed number out of range. Possible \
                        values: 0 to {0}".format(nruns-1)
                q = raw_input("Enter a new population number, or enter -1 \
                        to seed all populations, or skip to abort.")
                while q not in range(-1, nruns): 
                    # Repeat until valid input given
                    if q == "": exit("Aborting.")
                    q = raw_input("Please enter a valid integer from -1 \
                            to {0}, or skip to abort.".format(nruns-1))
                return self.get_startpop(seed, q)
            self.logprint(status.format("population {0}".format(pop_number)))
            self.startpop = [simobj.runs[pop_number].population]
        return

    def finalise(self, file_pref, log_pref):
        """Finish recording and save output files."""
        self.average_records()
        self.endtime = datetime.datetime.now()
        simend = time.strftime('%X %x', time.localtime())+"."
        self.logprint("\nSimulation completed at "+simend)
        self.logprint(get_runtime(self.starttime, self.endtime))
        self.logprint("Saving output and exiting.\n")
        sim_file = open(file_pref + ".sim", "wb")
        log_file = open(log_pref + ".txt", "w")
        rec_file = open(file_pref + ".rec", "wb")
        try:
            log_file.write(self.log)
            pickle.dump(self, sim_file)
            if hasattr(self, "avg_record"):
                pickle.dump(self.avg_record, rec_file)
            else:
                pickle.dump(self.runs[0].record, rec_file)
        finally:
            sim_file.close()
            log_file.close()
            rec_file.close()

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        print message
        self.log += message+"\n"

    def average_records(self):
        if len(self.runs) == 1: 
            self.avg_record = self.runs[0].record
            return
        self.avg_record = Record(self.conf)
        rec_list = [x.record for x in self.runs if x.complete and not x.dieoff]
        rec_list = [r.record for r in rec_list] # Get record dicts
        # First test that all runs are compatible
        eq_array_0 = np.array([[len(r["genmap"]), r["chr_len"], r["n_bases"],
            r["max_ls"], r["maturity"]] for r in rec_list])
        eq_array_1 = np.array([list(r["d_range"])+list(r["d_range"])+\
                list(r["snapshot_stages"]) for r in rec_list])
        eq_array_2 = np.array([r["death_mean"].shape+r["density_surv"].shape+\
                r["entropy"].shape+r["resources"].shape+r["n1"].shape+\
                r["s1"].shape+r["age_distribution"].shape for r in rec_list])
        cm = np.all(np.isclose(eq_array_0, eq_array_0[0])) and \
                np.all(np.isclose(eq_array_1, eq_array_1[0])) and \
                np.all(np.isclose(eq_array_2,eq_array_2[0]))
        if cm: # Test if record item dimensions are concordant
            sar,rr = self.avg_record.record, rec_list[0].keys()
            self.logprint("Runs are compatible; generating averaged data.")
            for key in rr:
                karray = np.array([r[key] for r in rec_list])
                sar[key],sar[key+"_SD"] = np.mean(karray, 0),np.std(karray, 0)
                if isinstance(sar[key],np.ndarray):
                    if np.all(np.isclose(sar[key], sar[key].astype(int))):
                            sar[key] = sar[key].astype(int)
                elif sar[key] == int(sar[key]):
                    sar[key] = int(sar[key])
            # Calculate number and % of failed runs explicitly
            sar["n_runs"] = len(self.runs)
            sar["n_successes"] = \
                    sum([x.complete and not x.dieoff for x in self.runs])
            sar["prev_failed"] = \
                np.sum([r.record.record["prev_failed"] for r in self.runs])
            sar["percent_dieoff"] = 100*\
                    (sar["prev_failed"]+sum([x.dieoff for x in self.runs]))/\
                    (sar["prev_failed"]+len(self.runs))
            self.avg_record.record
            return
        else:
            np.set_printoptions(threshold=np.inf)
            print eq_array_0-eq_array_0[0]
            print eq_array_1-eq_array_1[0]
            print eq_array_2-eq_array_2[0]
            raise ValueError("Cannot generate average run data;"+\
                    " runs incompatible.")
