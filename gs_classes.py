import gs_functions as fn
import numpy as np
import scipy.stats as st
from random import sample
from operator import mul

class Population:
    """A simulated population with genomes and ages."""

    # Initialisation
    def __init__(self, params, genmap, ages="", genomes=""):
        self.sex = params["sexual"]
        self.chrlen = params["chr_len"]
        self.nbase = params["n_base"]
        self.maxls = params["max_ls"]
        self.maturity = params["maturity"]
        self.genmap = genmap
        # Determine ages if not given
        if ages == "" and params["age_random"]:
            self.ages = np.random.random_integers(0, self.maxls-1,
                    params["start_pop"])
        elif ages == "":
            self.ages = np.repeat(self.maturity, params["start_pop"])
        else: self.ages = np.copy(ages)
        # Determine genomes if not given
        if genomes == "":
            self.genomes = fn.make_genome_array(
                    params["start_pop"], self.chrlen, self.genmap,
                    self.nbase, params["g_dist"])
        else: self.genomes = np.copy(genomes)
        self.N = len(self.ages)
        self.index = np.arange(self.N)

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

    def addto(self, pop):
        """Append the individuals from a second population to this one,
        keeping this one's parameters and genome map."""
        self.ages = np.append(self.ages, pop.ages)
        self.genomes = np.vstack([self.genomes, pop.genomes])

    # Major methods:

    def get_subpop(self, min_age, max_age, offset, val_range):
        """Select a population subset based on chance defined by genotype."""
        subpop_indices = np.empty(0,int)
        for age in range(min_age, min(max_age, max(self.ages)+1)):
            # Get indices of appropriate locus for that age:
            locus = np.nonzero(self.genmap==(age+offset))[0][0]
            pos = np.arange(locus*self.nbase, (locus+1)*self.nbase)
            # Subset to correct age and required locus:
            match = (self.ages == age)
            which = np.nonzero(match)[0]
            pop = self.genomes[match][:,np.append(pos, pos+self.chrlen)]
            # Get sum of genotype for every individual:
            gen = np.sum(pop, axis=1)
            # Convert genotypes into inclusion probabilities:
            inc_rates = val_range[gen]
            included = which[fn.chance(inc_rates, len(inc_rates))]
            subpop_indices = np.append(subpop_indices, included)
        subpop = Population(self.params(), self.genmap,
                self.ages[subpop_indices], self.genomes[subpop_indices])
        return subpop

    def growth(self, r_range, penf, r_rate,  m_rate, m_ratio, verbose):
        """Generate new mutated children from selected parents."""
        if verbose:
            fn.logprint("Calculating reproduction...", False)
        parents = self.get_subpop(self.maturity, self.maxls, 100, r_range/penf)
        if self.sex and parents.N > 1:
            parents.__recombine(r_rate)
            children = parents.__assortment()
        else: children = parents.clone()
        children.__mutate(m_rate, m_ratio)
        children.ages[:] = 0 # Make newborn
        self.addto(children)
        self.N = len(self.ages)
        if verbose:
            fn.logprint("done. ", False)
            fn.logprint(str(children.N)+" new individuals born.")

    def death(self, d_range, penf, verbose):
        """Select survivors and kill rest of population."""
        if verbose: fn.logprint("Calculating death...", False)
        val_range = 1-(d_range*penf)
        survivors = self.get_subpop(0, self.maxls, 0, val_range)
        if verbose:
            dead = self.N - survivors.N
            fn.logprint("done. "+str(dead)+" individuals died.")
        self.ages = survivors.ages
        self.genomes = survivors.genomes
        self.N = len(self.ages)

    def crisis(self, crisis_sv, n_stage):
        """Apply an extrinsic death crisis and subset population."""
        n_survivors = int(self.N*crisis_sv)
        which_survive = np.random.choice(self.index, n_survivors, False)
        self.ages = self.ages[which_survive]
        self.genomes = self.genomes[which_survive]
        self.N = len(self.ages)
        fn.logprint("Stage " + n_stage + ": Crisis! ", False)
        fn.logprint(str(n_survivors)+" individuals survived.")

    # Private methods:

    def __recombine(self, r_rate):
        """Recombine between the two chromosomes of each individual
        in the population."""
        if r_rate > 0:
            chr1 = np.arange(self.chrlen)
            chr2 = chr1 + self.chrlen
            for n in range(len(self.genomes)):
                g = self.genomes[n]
                r_sites = np.nonzero(fn.chance(r_rate, self.chrlen))[0]
                for r in r_sites:
                    g = np.concatenate((g[chr1][:r], g[chr2][r:],
                        g[chr2][:r], g[chr1][r:]))
                self.genomes[n] = g

    def __assortment(self):
        """Pair individuals into breeding pairs and generate children
        through random assortment."""
        pop = self.clone()
        # Must be even number of parents:
        if pop.N%2 != 0:
            ix = sample(range(pop.N), 1)
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

    def __mutate(self, m_rate, m_ratio):
        """Mutate genomes of population according to stated rates."""
        if m_rate > 0:
            is_0 = self.genomes==0
            is_1 = np.invert(is_0)
            positive_mut = fn.chance(m_ratio*m_rate, np.sum(is_0))
            negative_mut = 1-fn.chance(m_rate,np.sum(is_1))
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
            "junk_fitness":np.copy(array3)
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
        s0 = reduce(mul, p.genomes[:,:p.chrlen].shape) / s1
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
        """Sort array in ascending order by age (survival:0-71, 
        reproduction: 16-71, neutral). Array must have same number of
        columns as genome array."""
        b = self.record["n_bases"]
        m = self.record["maturity"]
        maxls = self.record["max_ls"]
        count = 0
        arr_sorted = np.zeros(arr.shape)
        for i in self.record["genmap"]:
            if i<100: # survival
                arr_sorted[range(i*b, (i+1)*b)] = arr[range(count, count+b)]
            elif i>=200: # neutral
                arr_sorted[-b:] = arr[range(count, count+b)]
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
        n1s = np.vstack((p.genomes[:, :p.chrlen], p.genomes[:, p.chrlen:]))
        n1_std = np.std(n1s, axis=0)
        n1 = np.mean(n1s, axis=0) # Mean number of 1's per chromosome bit
        # Junk stats calculated from neutral locus
        neut_locus = np.nonzero(p.genmap==201)[0][0]
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

    def update(self, population, resources, surv_penf, repr_penf, stage, n_snap):
        """Record detailed population data at current snapshot stage."""
        self.quick_update(stage, population, resources, surv_penf, repr_penf)
        self.update_agestats(population, n_snap)
        self.update_invstats(population, n_snap)

    def age_wise_n1(self, arr_str):
        """Average n1 array, starting from value-per-bit, for it to be value-per-age. [arr_str = 'n1' or 'n1_std']"""
        b = self.record["n_bases"]
        arr = self.record[arr_str] # already sorted
        s = arr.shape
        res = np.mean(arr.reshape((s[0], self.record["chr_len"]/b, b)), 2)
        return res
    def final_update(self, n_run, window):
        """After run completion, compute fitness and s1 (rolling window std of n1)."""
        # Rolling standard deviation of #1's along genome:
        a = self.record["n1"]
        a_shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        a_strd = a.strides + (a.strides[-1],) # strides
        self.record["s1"] = np.lib.stride_tricks.as_strided(a,
                shape=a_shape, strides=a_strd)
        x_surv = np.cumprod(1-self.record["death_mean"],1)
        self.record["fitness"] = np.cumsum(
                x_surv*self.record["repr_mean"],1)
        self.record["junk_fitness"] = (
                1-self.record["junk_death"])*self.record["junk_repr"]
        self.record["actual_death_rate"] = self.actual_death_rate()
        self.record["age_wise_n1"] = self.age_wise_n1("n1")
        self.record["age_wise_n1_std"] = self.age_wise_n1("n1_std")

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
