########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Population, Outpop                                          #
# Description: A basic simulated population, with ages, genomes,       #
#   generations, and growth and death methods.                         #
########################################################################

## PACKAGE IMPORT ##
from .functions import chance, init_ages, init_genomes, init_generations
from .Config import deepeq
import numba
cimport numba
import numpy as np
cimport numpy as np
import random, copy

## CYTHON SETUP ##
# cython: profile=True
NPINT = np.int # Integer arrays
NPFLOAT = np.float # Decimal arrays
NPBOOL = np.uint8 # Boolean arrays
ctypedef np.int_t NPINT_t
ctypedef np.float_t NPFLOAT_t
ctypedef np.uint8_t NPBOOL_t

# TODO: Remove maximum lifespan
# TODO: Add init test for correct params keys
# TODO: Remove age_random from config

cdef class Population:
    """A simulated population with genomes, ages and generation numbers,
    capable of undergoing growth and death."""
    cdef public np.ndarray genmap, ages, genomes, genmap_argsort, generations
    cdef public int recombine, assort, chr_len, n_base, max_ls, maturity, N
    cdef public int repr_offset, neut_offset
    cdef public str repr_mode
    cdef public dict g_dist
    #! TODO: Specify array dimensions?

    ## INITIALISATION ##

    def __init__(self, dict params, 
            np.ndarray[NPINT_t, ndim=1] genmap,
            np.ndarray[NPINT_t, ndim=1] ages,
            np.ndarray[NPINT_t, ndim=2] genomes,
            np.ndarray[NPINT_t, ndim=1] generations,
            ):
        """Create a new population, either with newly-generated age and genome
        vectors or inheriting these from a seed."""
        self.set_genmap(genmap) # Define genome map
        self.set_attributes(params) # Define population parameters
        self.set_initial_size(params, ages, genomes, generations) # Define size
        self.fill(ages, genomes, generations) # Generate individuals

    def set_genmap(self, np.ndarray[NPINT_t, ndim=1] genmap):
        """Set Population genome map from an input array."""
        self.genmap = np.copy(genmap) # Map of locus identities
        self.genmap_argsort = np.argsort(genmap) # Indices for sorted genmap

    def set_attributes(self, dict params):
        """Set Population attributes from a parameter dictionary."""
        # Set reproductive mode
        self.repr_mode = params["repr_mode"]
        repr_values = {"sexual":[1,1], "asexual":[0,0],
                "recombine_only":[1,0], "assort_only":[0,1]}
        self.recombine, self.assort = repr_values[self.repr_mode]
        # Set other attributes
        self.chr_len = params["chr_len"] # Number of bits per chromosome
        self.maturity = params["maturity"] # Age of maturity (for reproduction)
        self.max_ls = params["max_ls"] # Maximum lifespan
        self.n_base = params["n_base"] # Number of bits per locus
        self.g_dist = params["g_dist"].copy() # Proportions of 1's in initial loci
        self.repr_offset = params["repr_offset"] # Genmap offset for repr loci
        self.neut_offset = params["neut_offset"] # Genmap offset for neut loci

    def set_initial_size(self, dict params, 
            np.ndarray[NPINT_t, ndim=1] ages,
            np.ndarray[NPINT_t, ndim=2] genomes,
            np.ndarray[NPINT_t, ndim=1] generations,
            ):
        """Determine population size from initial inputs."""
        new_ages = np.array_equal(ages, init_ages())
        new_genomes = np.array_equal(genomes, init_genomes())
        new_generations = np.array_equal(generations, init_generations())
        if not new_ages:
            if not new_genomes and len(ages) != len(genomes):
                errstr = "Size mismatch between age and genome arrays."
                raise ValueError(errstr)
            if not new_generations and len(ages) != len(generations):
                errstr = "Size mismatch between age and generation arrays."
                raise ValueError(errstr)
            self.N = len(ages)
        elif not new_genomes:
            if not new_generations and len(genomes) != len(generations):
                errstr = "Size mismatch between genome and generation arrays."
                raise ValueError(errstr)
            self.N = len(genomes)
        elif not new_generations:
            self.N = len(generations)
        else:
            self.N = params["start_pop"]

    def fill(self, 
            np.ndarray[NPINT_t, ndim=1] ages,
            np.ndarray[NPINT_t, ndim=2] genomes,
            np.ndarray[NPINT_t, ndim=1] generations,
            ):
        """Fill a new Population object with individuals based on input
        age, genome and generation arrays."""
        # Test for new vs seeded values
        new_ages = np.array_equal(ages, init_ages())
        new_genomes = np.array_equal(genomes, init_genomes())
        new_generations = np.array_equal(generations, init_generations())
        # Specify individual value arrays
        if new_ages:
            ages = np.random.randint(0,self.max_ls-1,self.N)
        if new_genomes:
            genomes = self.make_genome_array()
        if new_generations:
            generations = np.repeat(0L, self.N)
        self.ages = np.copy(ages)
        self.genomes = np.copy(genomes)
        self.generations = np.copy(generations)

    def make_genome_array(self):
        """Generate initial genomes for start_pop individuals according to
        population parameters, with uniformly-distributed bit values with
        proportions specified by self.g_dist."""
        # Initialise genome array
        genome_array = np.zeros([self.N, self.chr_len*2])
        # Use genome map to determine probability distribution for each locus:
        loci = {
            "s":np.nonzero(self.genmap<self.repr_offset)[0],
            "r":np.nonzero(np.logical_and(self.genmap>=self.repr_offset,
                self.genmap<self.neut_offset))[0],
            "n":np.nonzero(self.genmap>=self.neut_offset)[0]
            }
        # Set genome array values according to given probabilities:
        for k in loci.keys():
            # Identify genome positions corresponding to locus type
            pos = np.array([range(self.n_base) + x for x in loci[k]*self.n_base])
            pos = np.append(pos, pos + self.chr_len)
            # Add values to positions according to appropriate distribution
            genome_array[:, pos] = chance(self.g_dist[k], [self.N, len(pos)])
        return genome_array.astype(int)


    ## REARRANGEMENT AND COMBINATION ## #! TODO: Cythonise these methods?

    def params(self):
        """Get population-initiation parameters from present population."""
        p_dict = {
                "repr_mode":self.repr_mode,
                "chr_len":self.chr_len,
                "n_base":self.n_base,
                "max_ls":self.max_ls,
                "maturity":self.maturity,
                "g_dist":self.g_dist,
                "repr_offset":self.repr_offset,
                "neut_offset":self.neut_offset
                }
        return p_dict

    def clone(self):
        """Generate a new, identical population object."""
        return Population(self.params(), self.genmap,
                self.ages, self.genomes, self.generations)

    def attrib_rep(self, function, pop2=""):
        """Repeat an operation over all relevant attributes, then update N."""
        for attr in ["ages","genomes","generations"]:
            val1,val2 = getattr(self,attr), getattr(pop2,attr) if pop2 else ""
            setattr(self,attr,function(val1) if val2=="" else function(val1,val2))
        self.N = len(self.ages)

    def shuffle(self):
        """Rearrange order of individuals in population."""
        index = np.arange(self.N)
        np.random.shuffle(index)
        def f(x): return x[index]
        self.attrib_rep(f)

    def subset_members(self,targets):
        """Keep a subset of members and discard the rest, according to an
        index of booleans (True = keep)."""
        def f(x): return x[targets]
        self.attrib_rep(f)

    def subtract_members(self,targets):
        """Remove members from population, according to an index of integers."""
        def f(x): return np.delete(x,targets, 0)
        self.attrib_rep(f)

    def add_members(self, pop):
        """Append the individuals from a second population to this one,
        keeping this one's parameters and genome map."""
        def f(x,y): return np.concatenate((x,y), 0)
        self.attrib_rep(f, pop)

    def subset_clone(self, targets):
        """Create a clone population and subset its members."""
        pop = self.clone()
        pop.subset_members(targets)
        return pop

    ## INCREMENT VALUES ##

    def increment_ages(self):
        """Age all individuals in population by one stage."""
        self.ages += 1L

    def increment_generations(self):
        """Increase generation of all individuals in population by one."""
        self.generations += 1L

    ## CHROMOSOMES AND LOCI ##

    cpdef chrs(self, int reshape):
        """Return an array containing the first and second chromosomes 
        of each individual in the population, in either 2D individual/bit
        or 3D individual/locus/bit configuration."""
        if not reshape:
            # [chromosome, individual, bit]
            return self.genomes.reshape((self.N,2,self.chr_len)
                    ).transpose((1,0,2))
        else:
            # [chromosome, individual, locus, bit]
            return self.genomes.reshape((self.N,2,len(self.genmap),self.n_base)
                    ).transpose(1,0,2,3)
        #! Not happy with the transposition efficiency-wise, but having 
        # individuals first screws up recombination/assortment in ways I
        # don't know how to efficiently fix...

    cpdef sorted_loci(self):
        """Return the sorted locus genotypes of the individuals in the 
        population, summed within each locus and across chromosomes."""
        cdef:
            np.ndarray[NPINT_t, ndim=2] locs
            np.ndarray[NPINT_t, ndim=4] chrs # NOTE shouldn't this be chrx?
        # Get chromosomes of population, arranged by locus
        chrx = self.chrs(True) 
        # Collapse bits into locus sums and add chromosome values together
        # to get total genotype value for each locus (dim1=indiv, dim2=locus)
        locs = np.einsum("ijkl->jk", chrx)
        return locs[:,self.genmap_argsort]

    cpdef surv_loci(self):
        """Return the sorted survival locus genotypes of each individual in
        the population."""
        return self.sorted_loci()[:,:self.max_ls]

    cpdef repr_loci(self):
        """Return the sorted reproduction locus genotypes of each individual in
        the population."""
        return self.sorted_loci()[:,self.max_ls:(2*self.max_ls-self.maturity)]

    cpdef neut_loci(self):
        """Return the sorted neutral locus genotypes of each individual in
        the population."""
        return self.sorted_loci()[:,(2*self.max_ls-self.maturity):]

    #! TODO: Fix the above 3 functions to use internal index information,
    #  rather than assuming a particular hard-coded order

    ## GROWTH AND DEATH ##

    cpdef get_subpop_old(self, int min_age, int max_age, # Ages of indivs to use
        int offset, # Offset value to relate ages to genmap values
        np.ndarray[NPFLOAT_t, ndim=1] val_range): # Genotype:probability map
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
        # Reshape genome array: [individuals,loci,bits]
        genloc = np.reshape(self.genomes, (self.N, g*2, self.n_base))
        # Get inclusion probabilities age-wise:
        for age in range(min_age, min(max_age, np.max(self.ages)+1)):
            # Get indices of appropriate locus for that age:
            locus = np.ndarray.nonzero(self.genmap==(age+offset))[0][0]
            # Subset to genome positions in required locus, on both chromosomes,
            # for individuals of correct age
            which = np.nonzero(self.ages == age)[0]
            pop = genloc[which][:,[locus, locus+g]]
            # Determine inclusion probabilities for these individuals based on
            # their locus genotype and the value range given
            inc_probs[which] = val_range[np.einsum("ijk->i", pop)]
        inc = chance(inc_probs, self.N)
        return inc # Binary array giving inclusion status of each individual
    #! TODO: Benchmark old vs new get_subpop functions

    cpdef get_subpop(self, 
            np.ndarray[NPINT_t, ndim=1] age_bounds, # Age cohorts to consider
            np.ndarray[NPINT_t, ndim=2] genotypes, # Relevant per-age GT sums
            np.ndarray[NPFLOAT_t, ndim=1] val_range): # GT:probability map
        # Get age array for suitably-aged individuals
        ages = np.clip(self.ages - age_bounds[0], 0, np.diff(age_bounds)-1)
        genotypes = genotypes # NOTE this is obsolete
        # Get relevant genotype sum for each individual of appropriate age
        # NOTE len(age)=pop.N, so why not just use that?
        gt = genotypes[np.arange(len(ages)), ages]
        # Convert to inclusion probabilities and compute inclusion
        inc_probs = val_range[gt]
        subpop = chance(inc_probs, self.N) # Binary array of inclusion statuses
        # Subset by age boundaries and return
        # COMMENT: we can't subset by age boundaries before this point because
        # we need the output to be of length pop.N
        inc = np.logical_and(self.ages>=age_bounds[0],
                self.ages<age_bounds[1])
        return subpop * inc

    cpdef death(self, 
            np.ndarray[NPFLOAT_t, ndim=1] s_range, # Survival probabilities
            float penf): # Starvation penalty factor
        """Select survivors and kill rest of population."""
        cdef:
            np.ndarray[NPFLOAT_t, ndim=1] d_range
            np.ndarray[NPINT_t, ndim=1] age_bounds
            np.ndarray[NPBOOL_t, ndim=1,cast=True] survivors
        if self.N == 0: return # If no individuals in population, do nothing
        # Get inclusion probabilities
        d_range = np.clip((1-s_range)*penf, 0, 1) # Death probs (0 to 1 only)
        s_range = 1-d_range # Convert back to survival again
        age_bounds = np.array([0,self.max_ls])
        # Identify survivors and remove others
        survivors = self.get_subpop(age_bounds, self.surv_loci(), s_range)
        self.subset_members(survivors)

    cpdef make_children(self, 
            np.ndarray[NPFLOAT_t, ndim=1] r_range, # Reprodn probabilities
            float penf, # Starvation penalty factor
            float m_rate, # Per-bit mutation rate
            float m_ratio, # Positive/negative mutation ratio
            float r_rate): # Recombination rate (if recombining)
        """Generate new mutated children from selected parents."""
        cdef:
            np.ndarray[NPBOOL_t, ndim=1,cast=True] which_parents
            np.ndarray[NPINT_t, ndim=1] age_bounds
            object parents, children
        if self.N == 0:# If no individuals in population, do nothing
            return self.subset_clone(np.zeros(self.N).astype(bool))
        # NOTE this is obsolete since x in (0,1) implies x/y in (0,1) for all y > 1
        # meaning if everything else well written and input ok, then obsolete
        r_range = np.clip(r_range / penf, 0, 1) # Limit to real probabilities
        age_bounds = np.array([self.maturity,self.max_ls])
        parents = self.get_subpop(age_bounds, self.repr_loci(), r_range)
        # Get children from parents
        if self.assort and np.sum(parents) == 1: # Need 2 parents here
            return self.subset_clone(np.zeros(self.N).astype(bool))
        children = self.subset_clone(parents)
        if self.recombine: children.recombination(r_rate)
        if self.assort: children.assortment()
        # Mutate children and add to population
        children.mutate(m_rate, m_ratio) 
        children.increment_generations()
        children.ages[:] = 0L # Make newborn
        return children

    cpdef growth(self,
            np.ndarray[NPFLOAT_t, ndim=1] r_range, # Reprodn probabilities
            float penf, # Starvation penalty factor
            float m_rate, # Per-bit mutation rate
            float m_ratio, # Positive/negative mutation ratio
            float r_rate): # Recombination rate (if recombining)
        """Generate new mutated children from selected parents."""
        cdef object children
        children = self.make_children(r_range, penf, m_rate, m_ratio, r_rate)
        self.add_members(children)

    cpdef mutate(self, float m_rate, # Per-bit mutation rate
            float m_ratio): # Positive/negative mutation ratio
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

    cpdef recombination(self, float r_rate): # Per-bit recombination rate
        """Recombine between the two chromosomes of each individual
        in the population."""
        cdef:
            np.ndarray[NPINT_t, ndim=2] r_sites, which_chr
            np.ndarray[NPINT_t, ndim=3] chrs
        if (r_rate > 0): # Otherwise do nothing
            # Randomly generate recombination sites
            r_sites = chance(r_rate, [self.N, self.chr_len]).astype(int)
            # Convert into [1,-1]
            r_sites = np.array([1,-1])[r_sites]
            # Determine crossover status of each position 
            # (1 = no crossover, -1 = crossover)
            which_chr = np.cumprod(r_sites, 1)
            # Convert to 0 = no crossover, 1 = crossover
            which_chr = 1 - (which_chr+1)/2
            # Generate new chromosomes and update genomes
            chrs = np.copy(self.chrs(0))
            self.genomes[:,:self.chr_len] = np.choose(which_chr, chrs)
            self.genomes[:,self.chr_len:] = np.choose(which_chr, chrs[[1,0]])

    def assortment(self):
        """Pair individuals into breeding pairs and generate children
        through random assortment."""
        if self.N == 1:
            raise ValueError("Cannot perform assortment with a single parent.")
        if self.N % 2 != 0: # If odd number of parents, discard one at random
            index = random.sample(range(self.N), 1)
            self.subtract_members(index)
        self.shuffle() # Randomly assign mating partners
        # Randomly combine parental chromatids
        which_pair = np.arange(self.N/2)*2 # First of each pair (0,2,4,...)
        # NOTE this is shuffling the second time? we can leave out this chance call?
        # could just do:
        # parent_0 = np.arange(self.N/2)*2
        # parent_1 = parent_0 + 1
        which_partner = chance(0.5,self.N/2)*1 # Member within pair (0 or 1)
        parent_0 = which_pair + which_partner # Parent 0
        parent_1 = which_pair + (1-which_partner) # Parent 1
        which_chr_0 = chance(0.5,self.N/2)*1 # Chromosome from parent 0
        which_chr_1 = chance(0.5,self.N/2)*1 # Chromosome from parent 1
        # Update population chromosomes
        chrs = np.copy(self.chrs(False))
        self.genomes[::2,:self.chr_len] = chrs[which_chr_0, parent_0]
        self.genomes[::2,self.chr_len:] = chrs[which_chr_1, parent_1]
        self.subset_members(np.tile([True,False], self.N/2))
 
    # Startpop method

    def __startpop__(self, pop_number):
        return Outpop(self).__startpop__(pop_number)

class Outpop:
    """Non-cythonised, pickle-able I/O form of Population class."""
    def __init__(self, pop):
        """Generate an Outpop from a Population object."""
        self.repr_mode = pop.repr_mode
        self.chr_len = pop.chr_len
        self.n_base = pop.n_base
        self.max_ls = pop.max_ls
        self.maturity = pop.maturity
        self.g_dist = pop.g_dist
        self.repr_offset = pop.repr_offset
        self.neut_offset = pop.neut_offset
        self.genmap = np.copy(pop.genmap)
        self.ages = np.copy(pop.ages)
        self.genomes = np.copy(pop.genomes)
        self.generations = np.copy(pop.generations)
        self.N = pop.N

    def params(self):
        """Get population-initiation parameters from present population."""
        p_dict = {
                "repr_mode":self.repr_mode,
                "chr_len":self.chr_len,
                "n_base":self.n_base,
                "max_ls":self.max_ls,
                "maturity":self.maturity,
                "g_dist":self.g_dist,
                "repr_offset":self.repr_offset,
                "neut_offset":self.neut_offset
                }
        return p_dict

    def toPop(self):
        """Make cythonised Population object from this Outpop."""
        return Population(self.params(), self.genmap, self.ages, 
                self.genomes, self.generations)

    def clone(self):
        """Generate a new, identical Outpop object."""
        return Outpop(self)

    # Comparison methods

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return NotImplemented
        return np.array_equal(self.genomes, other.genomes) and \
                np.array_equal(self.ages, other.ages) and \
                np.array_equal(self.generations, other.generations) and \
                deepeq(self.params(), other.params())
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented
    def __hash__(self):
        return hash(tuple(self.ages, self.genomes, self.generations, 
            self.params()))

    # Startpop method

    def __startpop__(self, pop_number):
        msg = "Setting seed directly from imported population."
        pop = self
        return (pop, msg)
