# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

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

def execute_run(run, keep_only_successes=False):
    if keep_only_successes: blank_run = copy.deepcopy(run)
    run.execute()
    if keep_only_successes and run.dieoff:
        run.logprint("Run failed. Total failures = {10. Repeating..."\
                .format(run.record.record["prev_failed"]))
        blank_run.record.record["prev_failed"] += 1
        blank_run.log = run.log + "\n"
        blank_run.starttime = run.starttime
        return execute_run(blank_run, True)
    run.logprint(get_runtime(run.starttime, run.endtime))
    return run

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
        self.sexual = c.sexual
        self.number_of_runs = c.number_of_runs
        self.number_of_stages = c.number_of_stages
        self.crisis_p = c.crisis_p
        self.crisis_stages = c.crisis_stages
        self.crisis_sv = c.crisis_sv
        self.number_of_snapshots = c.number_of_snapshots
        self.res_start = c.res_start
        self.res_var = c.res_var
        self.res_limit = c.res_limit
        self.R = c.R
        self.V = c.V
        self.start_pop = c.start_pop
        self.age_random = c.age_random
        self.g_dist_s = c.g_dist_s
        self.g_dist_r = c.g_dist_r
        self.g_dist_n = c.g_dist_n
        self.death_bound = c.death_bound
        self.repr_bound = c.repr_bound
        self.r_rate = c.r_rate
        self.m_rate = c.m_rate
        self.m_ratio = c.m_ratio
        self.max_ls = c.max_ls
        self.maturity = c.maturity
        self.n_neutral = c.n_neutral
        self.n_base = c.n_base
        self.surv_pen = c.surv_pen
        self.repr_pen = c.repr_pen
        self.death_inc = c.death_inc
        self.repr_dec = c.repr_dec
        self.window_size = c.window_size

    def generate(self):
        """Generate derived configuration parameters from simple ones and
        add to configuration object."""
        self.g_dist = {
                "s":self.g_dist_s, # Survival
                "r":self.g_dist_r, # Reproduction
                "n":self.g_dist_n # Neutral
                }
        # Genome map: survival (0 to max), reproduction (maturity to max), neutral
        self.genmap = np.asarray(range(0,self.max_ls) +\
                range(self.maturity+100,self.max_ls+100) +\
                range(200, 200+self.n_neutral))
        if self.sexual: self.repr_bound[1] *= 2
        # Length of chromosome in binary units
        self.chr_len = len(self.genmap) * self.n_base 
        # Probability ranges
        nstates = 2*self.n_base+1
        self.d_range = np.linspace(self.death_bound[1], 
                self.death_bound[0], nstates) # max to min death rate
        self.r_range = np.linspace(self.repr_bound[0],
                self.repr_bound[1], nstates) # min to max repr rate
        # Determine snapshot stages:
        if type(self.number_of_snapshots) is float:
            self.snapshot_proportion = self.number_of_snapshots
            self.number_of_snapshots = int(\
                    self.number_of_snapshots*self.number_of_stages)
        self.snapshot_stages = np.around(np.linspace(0,
            self.number_of_stages-1,self.number_of_snapshots),0)
        # Dictionary for population initialisation
        self.params = {"sexual":self.sexual, 
                "chr_len":self.chr_len, "n_base":self.n_base,
                "maturity":self.maturity, "max_ls":self.max_ls, 
                "age_random":self.age_random, 
                "start_pop":self.start_pop, "g_dist":self.g_dist}

class Outpop:
    """Pickle-able output form of Population class."""
    def __init__(self, pop):
        self.sex = pop.sex
        self.nbase = pop.nbase
        self.chrlen = pop.chrlen
        self.maxls = pop.maxls
        self.maturity = pop.maturity
        self.genmap = copy.copy(pop.genmap)
        self.ages = np.copy(pop.ages)
        self.genomes = np.copy(pop.genomes)
        self.N = pop.N

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

    def toPop(self):
        """Make cythonised Population object from this object."""
        return Population(self.params(), self.genmap, self.ages, 
                self.genomes)

    def clone(self):
        """Generate a new, identical population object."""
        return Outpop(self)

cdef class Population:
    """A simulated population with genomes and ages."""
    cdef public np.ndarray genmap, ages, genomes
    cdef public int sex, chrlen, nbase, maxls, maturity, N
    # Initialisation
    def __init__(self, dict params, np.ndarray[NPINT_t, ndim=1] genmap, 
            np.ndarray[NPINT_t, ndim=1] ages, 
            np.ndarray[NPINT_t, ndim=2] genomes):
        self.sex = params["sexual"] * 1
        self.nbase = params["n_base"]
        self.chrlen = params["chr_len"]
        self.maxls = params["max_ls"]
        self.maturity = params["maturity"]
        self.genmap = genmap
        # Determine ages and genomes if not given
        cdef np.ndarray[NPINT_t, ndim=1] testage = np.array([-1])
        cdef np.ndarray[NPINT_t, ndim=2] testgen = np.array([[-1],[-1]])
        if params.has_key("start_pop"):
            if np.shape(ages) == np.shape(testage) and \
                    (ages == testage).all():
                ages = np.random.random_integers(0, self.maxls-1,
                        params["start_pop"]) if params["age_random"]\
                        else np.repeat(self.maturity,params["start_pop"])
            if np.shape(genomes) == np.shape(testgen) and \
                    (genomes == testgen).all():
                genomes = self.make_genome_array(
                        params["start_pop"], params["g_dist"])
        self.ages = ages[:]
        self.genomes = genomes[:]
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
                np.copy(self.ages), np.copy(self.genomes))

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
            pos = np.array([range(self.nbase) + x for x in loci[k]*self.nbase])
            pos = np.append(pos, pos + self.chrlen)
            genome_array[:, pos] = chance(g_dist[k], [start_pop, len(pos)])
        return genome_array.astype("int")

    cpdef get_subpop(self, int min_age, int max_age, int offset,
            np.ndarray[NPFLOAT_t, ndim=1] val_range):
        """Select a population subset based on chance defined by genotype."""
        cdef:
            np.ndarray[NPINT_t, ndim=1] subpop_indices = np.empty(0,int)
            int age, locus, g
            np.ndarray[NPINT_t, ndim=1] pos, which
            np.ndarray[NPFLOAT_t, ndim=1] inc_rates
            np.ndarray[NPINT_t, ndim=3] genloc, pop
            np.ndarray[NPBOOL_t, ndim=1,cast=True] inc
        g = len(self.genmap)
        inc_rates = np.zeros(self.N)
        genloc = np.reshape(self.genomes, (self.N, g*2, self.nbase))
        # Get inclusion probabilities age-wise:
        for age in range(min_age, min(max_age, np.max(self.ages)+1)):
            # Get indices of appropriate locus for that age:
            locus = np.ndarray.nonzero(self.genmap==(age+offset))[0][0]
                # NB: Will only return FIRST locus for that age
            # Subset to correct age and required locus:
            which = np.nonzero(self.ages == age)[0]
            pop = genloc[which][:,[locus, locus+g]]
            # Determine inclusion rates
            inc_rates[which] = val_range[np.einsum("ijk->i", pop)]
        inc = chance(inc_rates, self.N)
        return inc

    cpdef growth(self, np.ndarray[NPFLOAT_t, ndim=1] var_range, float penf, 
            float r_rate, float m_rate, float m_ratio):
        """Generate new mutated children from selected parents."""
        cdef:
            np.ndarray[NPBOOL_t, ndim=1,cast=True] which_parents
            object parents, children
        if self.N == 0: return # Insulate from empty population
        r_range = np.clip(var_range / penf, 0, 1) # Limit to real probabilities
        which_parents = self.get_subpop(self.maturity, self.maxls, 100, 
                r_range/penf)
        parents = Population(self.params(), self.genmap,
                self.ages[which_parents], self.genomes[which_parents])
        if self.sex:
            if parents.N == 1:
                return # No children if only one parent
            else:
                parents.recombine(r_rate)
                children = parents.assortment()
        else: children = parents.clone()
        children.mutate(m_rate, m_ratio)
        children.ages[:] = 0 # Make newborn
        self.addto(children)
        self.N = len(self.ages)

    cpdef death(self, np.ndarray[NPFLOAT_t, ndim=1] d_range, float penf):
        """Select survivors and kill rest of population."""
        cdef:
            np.ndarray[NPFLOAT_t, ndim=1] val_range
            np.ndarray[NPBOOL_t, ndim=1,cast=True] survivors
            int new_N, dead
        if self.N == 0: return # Insulate from empty population
        val_range = np.clip(1-(d_range*penf),0,1) # Limit to real probabilities
        survivors = self.get_subpop(0, self.maxls, 0, val_range)
        self.ages = self.ages[survivors]
        self.genomes = self.genomes[survivors]
        self.N = np.sum(survivors)

    def crisis(self, crisis_sv):
        """Apply an extrinsic death crisis and subset population."""
        if self.N == 0: return # Insulate from empty population
        n_survivors = int(self.N*crisis_sv)
        which_survive = np.random.choice(np.arange(self.N), n_survivors, False)
        self.ages = self.ages[which_survive]
        self.genomes = self.genomes[which_survive]
        self.N = len(self.ages)

    # Private methods:

    cpdef recombine(self, int r_rate):
        """Recombine between the two chromosomes of each individual
        in the population."""
        cdef:
            int n, r
            np.ndarray[NPINT_t, ndim=1] chr1, chr2, r_sites, g
        if (r_rate > 0):
            chr1 = np.arange(self.chrlen)
            chr2 = chr1 + self.chrlen
            for n in range(self.N):
                g = self.genomes[n]
                r_sites = np.nonzero(chance(r_rate, self.chrlen))[0]
                for r in r_sites:
                    g = np.concatenate((g[chr1][:r], g[chr2][r:],
                        g[chr2][:r], g[chr1][r:]))
                self.genomes[n] = g

    def assortment(self):
        """Pair individuals into breeding pairs and generate children
        through random assortment."""
        pop = self.clone()
        # Must be even number of parents:
        if pop.N%2 != 0:
            ix = random.sample(range(pop.N), 1)
            pop.genomes = np.delete(pop.genomes, ix, 0)
            pop.N -= 1
        # Randomly assign mating partners:
        pop.shuffle()
        # Randomly combine parental chromatids
        chr1 = np.arange(self.chrlen)
        chr2 = chr1 + self.chrlen
        chr_choice = np.random.choice(["chr1","chr2"], pop.N)
        chr_dict = {"chr1":chr1, "chr2":chr2}
        for m in range(pop.N/2):
            pop.genomes[2*m][chr_dict[chr_choice[2*m]]] = \
                pop.genomes[2*m+1][chr_dict[chr_choice[2*m+1]]]
        # Generate child population
        children = Population(pop.params(), pop.genmap,
                pop.ages[::2], pop.genomes[::2])
        return(children)

    cpdef mutate(self, float m_rate, float m_ratio):
        """Mutate genomes of population according to stated rates."""
        cdef:
            np.ndarray[NPBOOL_t, ndim=2,cast=True] is_0, is_1
            np.ndarray[NPINT_t, ndim=1] positive_mut, negative_mut
        if m_rate > 0:
            is_0 = self.genomes==0
            is_1 = np.invert(is_0)
            positive_mut = chance(m_rate*m_ratio, np.sum(is_0)).astype(int)
            negative_mut = 1-chance(m_rate, np.sum(is_1))
            self.genomes[is_0] = positive_mut
            self.genomes[is_1] = negative_mut

class Record:
    """An enhanced dictionary object recording simulation data."""

    def __init__(self, population, snapshot_stages, n_stages, d_range,
            r_range, window_size):
        """ Create a new dictionary object for recording output data."""
        m = len(snapshot_stages)
        array1 = np.zeros([m,population.maxls])
        array2 = np.zeros([m,2*population.nbase+1])
        array3 = np.zeros(m)
        array4 = np.zeros(n_stages)
        self.record = {
            # Population parameters:
            "genmap":population.genmap,
            "chr_len":np.array([population.chrlen]),
            "n_bases":np.array([population.nbase]),
            "max_ls":np.array([population.maxls]),
            "maturity":np.array([population.maturity]),
            # Death and reproduction chance ranges:
            "d_range":d_range,
            "r_range":r_range,
            "snapshot_stages":snapshot_stages+1,
            # Per-stage data:
            "population_size":np.copy(array4),
            "resources":np.copy(array4),
            "surv_penf":np.copy(array4),
            "repr_penf":np.copy(array4),
            "age_distribution":np.zeros([n_stages,population.maxls]),
            # Per-age data:
            "death_mean":np.copy(array1),
            "death_sd":np.copy(array1),
            "repr_mean":np.copy(array1),
            "repr_sd":np.copy(array1),
            "fitness":np.copy(array1),
            # Genotype data:
            "density_surv":np.copy(array2),
            "density_repr":np.copy(array2),
            "n1":np.zeros([m,population.chrlen]),
            "n1_std":np.zeros([m,population.chrlen]),
            "s1":np.zeros([m,population.chrlen-window_size+1]),
            # Simple per-snapshot data:
            "entropy":np.copy(array3),
            "junk_death":np.copy(array3),
            "junk_repr":np.copy(array3),
            "junk_fitness":np.copy(array3),
            # Run info
            "dieoff":False,
            "prev_failed":0,
            "percent_dieoff":0
            }

    def quick_update(self, n_stage, population, resources, surv_penf, repr_penf):
        """Record only population size, age distribution, resource and penalty data."""
        p = population
        self.record["population_size"][n_stage] = p.N
        self.record["resources"][n_stage] = resources
        self.record["surv_penf"][n_stage] = surv_penf
        self.record["repr_penf"][n_stage] = repr_penf
        agedist = np.bincount(p.ages, minlength = p.maxls) / float(p.N)
        self.record["age_distribution"][n_stage] = agedist

    def update_agestats(self, population, n_snap):
        """Record detailed per-age statistics of population at
        current snapshot stage: death rate, reproduction rate, genotype
        density."""
        p = population
        b = p.nbase # Number of bits per locus
        # Initialise objects:
        # Genotype sum distributions:
        density_surv = np.zeros((2*b+1,))
        density_repr = np.zeros((2*b+1,))
        # Mean death/repr rates by age:
        death_mean = np.zeros(p.maxls)
        repr_mean = np.zeros(p.maxls)
        # Death/repr rate SD by age:
        death_sd = np.zeros(p.maxls)
        repr_sd = np.zeros(p.maxls)
        # Loop over ages:
        pop = p.genomes
        for age in range(p.maxls):
            if len(pop) > 0:
                # Find loci and binary units:
                surv_locus = np.nonzero(p.genmap==age)[0][0]
                surv_pos = np.arange(surv_locus*b, (surv_locus+1)*b)
                # Subset array to relevant columns and find genotypes:
                surv_pop = pop[:,np.append(surv_pos, surv_pos+p.chrlen)]
                surv_gen = np.sum(surv_pop, axis=1)
                # Find death/reproduction rates:
                death_rates = self.record["d_range"][surv_gen]
                # Calculate statistics:
                death_mean[age] = np.mean(death_rates)
                death_sd[age] = np.std(death_rates)
                density_surv += np.bincount(surv_gen, minlength=2*b+1)
                if age>=p.maturity:
                    # Same for reproduction if they're adults
                    repr_locus = np.nonzero(p.genmap==(age+100))[0][0]
                    repr_pos = np.arange(repr_locus*b, (repr_locus+1)*b)
                    repr_pop = pop[:,np.append(repr_pos,
                        repr_pos+p.chrlen)]
                    repr_gen = np.sum(repr_pop, axis=1)
                    repr_rates = self.record["r_range"][repr_gen]
                    repr_mean[age] = np.mean(repr_rates)
                    repr_sd[age] = np.std(repr_rates)
                    density_repr += np.bincount(repr_gen, minlength=2*b+1)
        # Average densities (there are total N*maxls genetic units)
        density_surv /= float(p.N*p.maxls)
        density_repr /= float(p.N*(p.maxls-p.maturity))
        # Update record
        self.record["death_mean"][n_snap] = death_mean
        self.record["death_sd"][n_snap] = death_sd
        self.record["repr_mean"][n_snap] = repr_mean
        self.record["repr_sd"][n_snap] = repr_sd
        self.record["density_surv"][n_snap] = density_surv
        self.record["density_repr"][n_snap] = density_repr

    def update_shannon_weaver(self, population):
        """H =-sum(p_i*ln(p_i)), where p_i is the density of genotype i."""
        p = population
        b = p.nbase
        s1 = b
        s0 = reduce(operator.mul, p.genomes[:,:p.chrlen].shape) / s1
        # s0 gives the total number of loci in each chromosome across
        # all individuals
        var = np.hstack((p.genomes[:,:p.chrlen].reshape(s0,s1), \
                         p.genomes[:,p.chrlen:].reshape(s0,s1)))
        # Horizontally stack matching loci from paired chromosomes 
        # to get (nbase * 2) x (# loci over whole population) array
        density = np.bincount(np.sum(var, axis=1), minlength = 2*b+1)
        # Get density distribution of each genotype (from 0 to 20 1's).
        return st.entropy(density)

    def sort_by_age(self, arr):
        """Sort a one-row array in ascending order by age (survival:0-71, 
        reproduction: 16-71, neutral). Array must have same number of
        element as genome array has columns."""
        b = self.record["n_bases"]
        m = self.record["maturity"]
        maxls = self.record["max_ls"]
        count = 0
        arr_sorted = np.zeros(arr.shape)
        for i in self.record["genmap"]:
            if i<100: # survival
                arr_sorted[range(i*b, (i+1)*b)] = arr[range(count, count+b)]
            elif i>=200: # neutral
                arr_sorted[len(arr_sorted)-b:] = arr[range(count, count+b)]
            else: # reproduction
                arr_sorted[range(maxls*b+(i-100-m)*b, maxls*b+(i+1-100-m)*b)] \
                = arr[range(count, count+b)]
            count += b
        return arr_sorted

    def update_invstats(self, population, n_snap):
        """Record detailed cross-population statistics at current
        snapshot stage: distribution of 1's (n1), entropy, junk genome (not under
        selection) values."""
        p = population
        # Frequency of 1's at each position on chromosome and it's std:
        n1s = np.append(p.genomes[:, :p.chrlen], p.genomes[:, p.chrlen:], 0)
        n1_std = np.std(n1s, axis=0)
        n1 = np.mean(n1s, axis=0) # Mean number of 1's per chromosome bit
        # Junk stats calculated from neutral locus
        neut_locus = np.nonzero(p.genmap==200)[0][0]
        neut_pos = np.arange(neut_locus*p.nbase, (neut_locus+1)*p.nbase)
        neut_pop = p.genomes[:,np.append(neut_pos, neut_pos+p.chrlen)]
        neut_gen = np.sum(neut_pop, axis=1)
        junk_death = np.mean(self.record["d_range"][neut_gen])
        junk_repr = np.mean(self.record["r_range"][neut_gen])
        # Append record object
        self.record["n1"][n_snap] = self.sort_by_age(n1)
        self.record["n1_std"][n_snap] = self.sort_by_age(n1_std)
        self.record["entropy"][n_snap] = self.update_shannon_weaver(population)
        self.record["junk_death"][n_snap] = junk_death
        self.record["junk_repr"][n_snap] = junk_repr

    def update(self, population, resources, surv_penf, repr_penf, stage, n_snap,
           full_update):
        """Record detailed population data at current snapshot stage."""
        self.quick_update(stage, population, resources, surv_penf, repr_penf)
        if full_update:
            self.update_agestats(population, n_snap)
            self.update_invstats(population, n_snap)

    def age_wise_n1(self, arr_str):
        """Average n1 array, starting from value-per-bit, for it to be value-per-age. [arr_str = 'n1' or 'n1_std']"""
        b = self.record["n_bases"]
        arr = self.record[arr_str] # already sorted
        s = arr.shape
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
        """After run completion, compute fitness and s1 (rolling window std of n1)."""
        # Rolling standard deviation of #1's along genome:
        a = self.record["n1"]
        d,s = a.ndim-1, len(a.strides)-1
        a_shape = a.shape[:d] + (a.shape[d] - window + 1, window)
        a_strd = a.strides + (a.strides[s],) # strides
        self.record["s1"] = np.std(
                np.lib.stride_tricks.as_strided(a, shape=a_shape, strides=a_strd), 2)
        x_surv = np.cumprod(1-self.record["death_mean"],1)
        self.record["fitness"] = np.cumsum(
                x_surv*self.record["repr_mean"],1)
        self.record["junk_fitness"] = (
                1-self.record["junk_death"])*self.record["junk_repr"]
        self.record["actual_death_rate"] = self.actual_death_rate()
        self.record["age_wise_n1"] = self.age_wise_n1("n1")
        self.record["age_wise_n1_std"] = self.age_wise_n1("n1_std")

class Run:
    """An object representing a single run of a simulation."""
    def __init__(self, config, startpop, n_run, report_n, verbose):
        self.log = ""
        self.conf = config
        self.surv_penf = 1.0
        self.repr_penf = 1.0
        self.resources = self.conf.res_start
        self.genmap = np.copy(self.conf.genmap)
        np.random.shuffle(self.genmap)
        if startpop != "":
            self.population = startpop.clone()
            # Adopt from population: genmap, nbase, chrlen
            self.genmap = self.population.genmap
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
            self.population = Outpop(Population(self.conf.params, self.genmap,
                    np.array([-1]), np.array([[-1],[-1]])))
        self.n_stage = 0
        self.n_snap = 0
        self.n_run = n_run
        self.record = Record(self.population, self.conf.snapshot_stages,
                self.conf.number_of_stages, self.conf.d_range, 
                self.conf.r_range, self.conf.window_size)
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
            self.resources = min(max(new_res, 0), self.conf.res_limit)

    def starving(self):
        """Determine whether population is starving based on resource level."""
        if self.conf.res_var:
            return self.resources == 0
        else:
            return self.population.N > self.resources

    def update_starvation_factors(self):
        """Update starvation factors under starvation."""
        if self.starving():
            self.surv_penf *= self.conf.death_inc if self.conf.surv_pen else 1
            self.repr_penf *= self.conf.repr_dec if self.conf.repr_pen else 1
        else: self.surv_penf = self.repr_penf = 1.0

    def execute_stage(self):
        """Perform one stage of a simulation run and test for completion."""
        if not isinstance(self.population, Population):
            raise TypeError("Convert Outpop objects to Population before running execute_stage.")
        report_stage = (self.n_stage % self.report_n == 0)
        if report_stage:
            self.logprint("Population = {0}.".format(self.population.N))
        self.dieoff = self.population.N == 0
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
            self.population.death(self.conf.d_range, self.surv_penf) # change
            n2 = self.population.N
            if full_report: 
                self.logprint("Done. {0} individuals born, {1} died."\
                        .format(n1-n0,n1-n2))
            if self.n_stage in self.conf.crisis_stages or chance(self.conf.crisis_p):
                self.population.crisis(self.conf.crisis_sv)
                self.logprint("Crisis! {0} individuals died, {1} survived."\
                        .format(n2-self.population.N, self.population.N))
        # Update run status
        self.dieoff = self.record.record["dieoff"] = self.population.N == 0
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
        if f>1 and not self.dieoff: 
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

    def __init__(self, config_file, seed_file, seed_n, report_n, verbose):
        self.starttime = datetime.datetime.now()
        simstart = time.strftime('%X %x', time.localtime())+".\n"
        self.log = ""
        self.logprint("\nBeginning simulation at "+simstart)
        self.logprint("Working directory: "+os.getcwd())
        self.get_conf(config_file)
        self.conf.generate()
        self.get_startpop(seed_file, seed_n)
        self.report_n = report_n
        self.verbose = verbose
        self.logprint("Initialising runs...")
        if len(self.startpop) == 1:
            self.runs = [Run(copy.deepcopy(self.conf), self.startpop[0],
                n, self.report_n, self.verbose) \
                for n in xrange(self.conf.number_of_runs)]
        else:
            self.runs = [Run(copy.deepcopy(self.conf), self.startpop[n],
                n, self.report_n, self.verbose) \
                for n in xrange(self.conf.number_of_runs)]
        self.logprint("Runs initialised. Executing...\n")

    def execute(self, nproc=-1, keep_only_successes=False):
        """Execute all runs."""
        if nproc < 0: # Use all available cores
            pool = multiprocessing.Pool()
        else: # Use specifed number of cores
            pool = multiprocessing.Pool(nproc)
        lock = multiprocessing.Lock()
        lock.acquire()
        try:
            asyncruns = []
            for n in xrange(self.conf.number_of_runs):
                asyncruns+= [pool.apply_async(execute_run, [self.runs[n],
                    keep_only_successes])]
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

    def get_startpop(self, seed_name="", pop_number=-1):
        """Import any seed simulation (or return blank)."""
        if seed_name == "":
            self.logprint("Seed: None.")
            self.startpop = [""]
            return
        try:
            # Make sure includes extension (default "sim")
            seed_name += ".sim" if os.path.splitext(seed_name)[1] == "" else ""
            simfile = open(seed_name, "rb")
            simobj = pickle.load(simfile) # import simulation object
            nruns = len(simobj.runs)
            if pop_number == -1:
                # -1 = seed all populations to equivalent runs in new sim
                if nruns != self.conf.number_of_runs:
                    exit("Error: number of runs in seed file does not match\
                            current configuration.")
                self.logprint("Seed: {0}, all populations.".format(seed_name))
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
                    return self.get_startpop(seed_name, q)
                self.logprint("Seed: {0}, population {1}.".format(seed_name, 
                    pop_number))
                self.startpop = [simobj.runs[pop_number].population]
            return
        except IOError:
            print "No such seed file: " + seed_name
            q = raw_input(
                    "Enter correct path to seed file, or skip to abort: ")
            if q == "": exit("Aborting.")
            r = raw_input("Enter population seed number, or enter -1 to \
                    seed all populations, or skip to abort.")
            if r == "": exit("Aborting.")
            return self.get_startpop(q, r)

    def finalise(self, file_pref, log_pref):
        """Finish recording and save output files."""
        self.endtime = datetime.datetime.now()
        simend = time.strftime('%X %x', time.localtime())+"."
        self.logprint("\nSimulation completed at "+simend)
        self.logprint(get_runtime(self.starttime, self.endtime))
        self.logprint("Saving output and exiting.\n")
        sim_file = open(file_pref + ".sim", "wb")
        log_file = open(log_pref + ".txt", "w")
        try:
            log_file.write(self.log)
            pickle.dump(self, sim_file)
        finally:
            sim_file.close()
            log_file.close()

    def logprint(self, message):
        """Print message to stdout and save in log object."""
        print message
        self.log += message+"\n"
