########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Population, Outpop                                          #
# Description: A basic simulated population, with ages, genomes,       #
#   generations, and growth and death methods.                         #
########################################################################

## PACKAGE IMPORT ##
from .functions import chance
from .functions import init_ages, init_genomes, init_generations, init_gentimes
from .Config import deep_eq
import numpy as np
import copy

class Population:
    """A simulated population with genomes, ages and generation numbers,
    capable of undergoing growth and death."""

    ## INITIALISATION ##

    def __init__(self, params, genmap, mapping, ages, genomes, generations, gentimes, targets=0):
        """Create a new population, either with newly-generated age and genome
        vectors or inheriting these from a seed."""
        self.set_genmap(genmap) # Define genome map
        self.mapping = np.copy(mapping) # Set mapping
        self.set_attributes(params) # Define population parameters
        self.set_initial_size(params, ages, genomes, generations, gentimes) # Define size
        self.fill(ages, genomes, generations, gentimes, targets) # Generate individuals

    def set_genmap(self, genmap):
        """Set Population genome map from an input array."""
        self.genmap = np.copy(genmap) # Map of locus identities
        self.genmap_argsort = np.argsort(genmap) # Indices for sorted genmap

    def set_attributes(self, params):
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
        self.object_max_age = params["object_max_age"] # Maximum age this Object can have in stages
        self.prng = params["prng"] # Numpy pseudo-random number generator

    def set_initial_size(self, params, ages, genomes, generations, gentimes):
        """Determine population size from initial inputs."""
        new_ages = np.array_equal(ages, init_ages())
        new_genomes = np.array_equal(genomes, init_genomes())
        new_generations = np.array_equal(generations, init_generations())
        new_gentimes = np.array_equal(gentimes, init_gentimes())
        if not new_ages:
            if not new_genomes and len(ages) != len(genomes):
                errstr = "Size mismatch between age and genome arrays."
                raise ValueError(errstr)
            if not new_generations and len(ages) != len(generations):
                errstr = "Size mismatch between age and generation arrays."
                raise ValueError(errstr)
            if not new_gentimes and len(ages) != len(gentimes):
                errstr = "Size mismatch between age and gentime arrays."
                raise ValueError(errstr)
            self.N = len(ages)
        elif not new_genomes:
            if not new_generations and len(genomes) != len(generations):
                errstr = "Size mismatch between genome and generation arrays."
                raise ValueError(errstr)
            if not new_gentimes and len(genomes) != len(gentimes):
                errstr = "Size mismatch between genome and gentime arrays."
                raise ValueError(errstr)
            self.N = len(genomes)
        elif not new_generations:
            if not new_gentimes and len(generations) != len(gentimes):
                errstr = "Size mismatch between generation and gentime arrays."
                raise ValueError(errstr)
            self.N = len(generations)
        elif not new_gentimes:
            self.N = len(gentimes)
        else:
            self.N = params["start_pop"]

    def fill(self, ages, genomes, generations, gentimes, targets=0):
        """Fill a new Population object with individuals based on input
        age, genome and generation arrays."""
        # Check for new vs seeded values
        new_ages = np.array_equal(ages, init_ages())
        new_genomes = np.array_equal(genomes, init_genomes())
        new_generations = np.array_equal(generations, init_generations())
        new_gentimes = np.array_equal(gentimes, init_gentimes())
        if not isinstance(targets, np.ndarray): targets = np.arange(self.N)
        else: self.N = targets.sum()
        # Specify individual value arrays
        if new_ages:
            ages = self.prng.randint(0,self.max_ls-1,self.N)
            targets_ages = np.arange(self.N)
        else: targets_ages = targets
        if new_genomes:
            genomes = self.make_genome_array()
            targets_genomes = np.arange(self.N)
        else: targets_genomes = targets
        if new_generations:
            generations = np.repeat(0L, self.N)
            targets_generations = np.arange(self.N)
        else: targets_generations = targets
        if new_gentimes:
            gentimes = np.zeros((self.N,2)).astype(int) if self.assort else np.zeros(self.N).astype(int)
            targets_gentimes = np.arange(self.N)
        else: targets_gentimes = targets
        self.ages = np.copy(ages[targets_ages])
        self.genomes = np.copy(genomes[targets_genomes])
        self.generations = np.copy(generations[targets_generations])
        self.gentimes = np.copy(gentimes[targets_gentimes])
        self.loci = self.sorted_loci()

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
            genome_array[:, pos] = chance(self.g_dist[k], (self.N, len(pos)), self.prng)
        return genome_array.astype(int)


    ## REARRANGEMENT AND COMBINATION ##

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
                "neut_offset":self.neut_offset,
                "object_max_age":self.object_max_age,
                "prng":self.prng
                }
        return p_dict

    def clone(self):
        """Generate a new, identical population object."""
        return Population(self.params(), self.genmap, self.mapping,
                self.ages, self.genomes, self.generations, self.gentimes)

    def attrib_rep(self, function, pop2="",
            attrs=["ages", "genomes", "generations", "gentimes", "loci"]):
        """Repeat an operation over all relevant attributes, then update N."""
        for attr in attrs:
            val1,val2 = getattr(self,attr), getattr(pop2,attr) if pop2 else ""
            setattr(self,attr,function(val1) if val2=="" else function(val1,val2))
        self.N = len(self.ages)

    def shuffle(self):
        """Rearrange order of individuals in population."""
        index = np.arange(self.N)
        self.prng.shuffle(index)
        def f(x): return x[index]
        self.attrib_rep(f)

    def subtract_members(self,targets):
        """Remove members from population, according to an index of integers."""
        def f(x): return np.delete(x,targets, 0)
        self.attrib_rep(f)

    def subset_members(self,targets):
        """Keep a subset of members and discard the rest, according to an
        index of booleans (True = keep)."""
        def f(x): return x[targets]
        self.attrib_rep(f)

    def add_members(self, pop):
        """Append the individuals from a second population to this one,
        keeping this one's parameters and genome map."""
        def f(x,y): return np.concatenate((x,y), 0)
        self.attrib_rep(f, pop)

    def subset_clone(self, targets):
        """Create a clone population and subset its members."""
        return Population(self.params(), self.genmap, self.mapping, self.ages,
                self.genomes, self.generations, self.gentimes, targets)

    ## INCREMENT VALUES ##

    def increment_ages(self):
        """Age all individuals in population by one stage."""
        self.ages += 1L

    def increment_generations(self):
        """Increase generation of all individuals in population by one."""
        self.generations += 1L

    ## CHROMOSOMES AND LOCI ##

    def sorted_loci(self):
        """Return sorted decimal coding for genotypes of individuals in population.
        Returned array is of shape (self.N, len(self.genmap))."""
        g = self.genomes.reshape((self.N, 2, len(self.genmap), self.n_base))
        g = g.transpose(0,2,1,3).reshape((self.N,len(self.genmap),2*self.n_base))
        g = g.dot(1<<np.arange(2*self.n_base)[::-1])
        return self.mapping[g][:, self.genmap_argsort]

    def chrs(self, reshape):
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

    def surv_loci(self):
        """Return the sorted survival locus genotypes of each individual in
        the population."""
        return self.loci[:,:self.max_ls]

    def repr_loci(self):
        """Return the sorted reproduction locus genotypes of each individual in
        the population."""
        return self.loci[:,self.max_ls:(2*self.max_ls-self.maturity)]

    def neut_loci(self):
        """Return the sorted neutral locus genotypes of each individual in
        the population."""
        return self.loci[:,(2*self.max_ls-self.maturity):]

    ## GROWTH AND DEATH ##

    def get_subpop(self,
            age_bounds, # Age cohorts to consider
            genotypes, # Relevant per-age GT sums
            val_range): # GT:probability map
        # Get age array for suitably-aged individuals
        ages = np.clip(self.ages - age_bounds[0], 0, np.diff(age_bounds)-1)
        # Get relevant genotype sum for each individual of appropriate age
        gt = genotypes[np.arange(self.N), ages]
        # Convert to inclusion probabilities and compute inclusion
        inc_probs = val_range[gt]
        subpop = chance(inc_probs, self.N, self.prng) # Binary array of inclusion statuses
        # Subset by age boundaries and return
        # Can not subset by age boundaries before this point because
        # need the output to be of length pop.N
        inc = np.logical_and(self.ages>=age_bounds[0],
                self.ages<age_bounds[1])
        return subpop * inc

    def death(self,
            s_range): # Survival probabilities
        """Select survivors and kill rest of population."""
        if self.N == 0: return # If no individuals in population, do nothing
        s_range = np.clip(s_range, 0, 1) # Limit to real probabilities
        # Get inclusion probabilities
        age_bounds = np.array([0,self.max_ls])
        # Identify survivors and remove others
        survivors = self.get_subpop(age_bounds, self.surv_loci(), s_range)
        self.subset_members(survivors)

    def make_children(self,
            r_range, # Reprodn probabilities
            m_rate, # Per-bit mutation rate
            m_ratio, # Positive/negative mutation ratio
            r_rate): # Recombination rate (if recombining)
        """Generate new mutated children from selected parents."""
        if self.N == 0:# If no individuals in population, do nothing
            return self.subset_clone(np.zeros(self.N).astype(bool))
        r_range = np.clip(r_range, 0, 1) # Limit to real probabilities
        age_bounds = np.array([self.maturity,self.max_ls])
        parents = self.get_subpop(age_bounds, self.repr_loci(), r_range)
        # Get children from parents
        if self.assort and np.sum(parents) == 1: # Need 2 parents here
            return self.subset_clone(np.zeros(self.N).astype(bool))
        children = self.subset_clone(parents)
        parent_ages = np.copy(children.ages).astype(int)
        if self.recombine: children.recombination(r_rate)
        if self.assort: children.assortment()
        # Mutate children and add to population
        children.mutate(m_rate, m_ratio)
        children.increment_generations()
        children.gentimes = parent_ages[0:children.N*2].reshape((children.N,2)) if self.assort\
                else parent_ages
        children.ages[:] = 0L # Make newborn
        return children

    def growth(self,
            r_range, # Reprodn probabilities
            m_rate, # Per-bit mutation rate
            m_ratio, # Positive/negative mutation ratio
            r_rate): # Recombination rate (if recombining)
        """Generate new mutated children from selected parents."""
        children = self.make_children(r_range, m_rate, m_ratio, r_rate)
        self.add_members(children)

    def mutate(self, m_rate, # Per-bit mutation rate
            m_ratio): # Positive/negative mutation ratio
        """Mutate genomes of population according to stated rates."""
        if m_rate > 0: # Else do nothing
            is_0 = self.genomes==0
            is_1 = np.invert(is_0)
            positive_mut = chance(m_rate*m_ratio, np.sum(is_0), self.prng).astype(int)
            negative_mut = 1-chance(m_rate, np.sum(is_1), self.prng)
            self.genomes[is_0] = positive_mut
            self.genomes[is_1] = negative_mut
        self.loci = self.sorted_loci() # Renew loci (inc. recomb./assortment)

    def recombination(self, r_rate, r_sites_fwd="", r_sites_rev=""):
        """Recombine between the two chromosomes of each individual
        in the population."""
        if (r_rate > 0): # Otherwise do nothing
            if not isinstance(r_sites_fwd,np.ndarray) or \
                    not isinstance(r_sites_rev,np.ndarray):
                # Randomly generate forward and reverse recombination sites
                r_sites_fwd = chance(r_rate/2, (self.N, self.chr_len),
                        self.prng).astype(int)
                r_sites_rev = chance(r_rate/2, (self.N, self.chr_len),
                        self.prng).astype(int)
            # Convert into [1,-1]
            r_sites_fwd = np.array([1,-1])[r_sites_fwd]
            r_sites_rev = np.array([1,-1])[r_sites_rev]
            # Determine crossover status of each position
            # (1 = no crossover, -1 = crossover)
            which_chr_fwd = np.cumprod(r_sites_fwd, 1)
            which_chr_rev = np.fliplr(np.cumprod(np.fliplr(r_sites_rev), 1))
            # Combine forward and reverse crossovers
            which_chr = which_chr_fwd * which_chr_rev
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
        self.shuffle() # Randomly assign mating partners
        # If odd number of parents, discard one
        if self.N % 2 != 0: self.subtract_members(np.array([self.N-1]))
        # Randomly combine parental chromatids
        parent_0 = np.arange(self.N/2)*2    # Parent 0
        parent_1 = parent_0 + 1             # Parent 1
        which_chr_0 = chance(0.5,self.N/2, self.prng)*1 # Chromosome from parent 0
        which_chr_1 = chance(0.5,self.N/2, self.prng)*1 # Chromosome from parent 1
        # Update population chromosomes
        chrs = np.copy(self.chrs(False))
        self.genomes[::2,:self.chr_len] = chrs[which_chr_0, parent_0]
        self.genomes[::2,self.chr_len:] = chrs[which_chr_1, parent_1]
        # Update generations as the max of two parents
        self.generations[::2] = np.maximum(self.generations[::2],
                self.generations[1::2])
        self.subset_members(np.tile([True,False], self.N/2))

    # Comparison methods

    def __eq__(self, other, prng_id=True):
        if not isinstance(other, self.__class__): return NotImplemented
        return np.array_equal(self.genomes, other.genomes) and \
                np.array_equal(self.ages, other.ages) and \
                np.array_equal(self.generations, other.generations) and \
                np.array_equal(self.gentimes, other.gentimes) and \
                deep_eq(self.params(), other.params(), prng_id=prng_id)
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented
    def __hash__(self):
        return hash(tuple(self.ages, self.genomes, self.generations,
            self.gentimes, self.params()))
