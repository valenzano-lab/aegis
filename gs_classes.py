import gs_functions as fn
import numpy as np
import scipy.stats as st
from random import sample

class Population:
    """A simulated population with genomes and ages."""

    # Initialisation
    def __init__(self, params, gen_map, ages="", genomes=""):
        self.sex = params["sexual"]
        self.chrlen = params["chr_len"]
        self.nbase = params["n_base"]
        self.maxls = params["max_ls"]
        self.maturity = params["maturity"]
        self.genmap = gen_map
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
        """Randomly select a population subset based on genotype."""
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

    def growth(self, r_range, r_rate, m_rate, m_ratio, verbose):
        """Generate new mutated children from selected parents."""
        if verbose: 
            fn.logprint("Calculating reproduction...", False)
        parents = self.get_subpop(self.maturity, self.maxls, 100, r_range)
        if self.sex:
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

    def death(self, d_range, starvation_factor, verbose):
        """Select survivors and kill rest of population."""
        if verbose: fn.logprint("Calculating death...", False)
        val_range = 1-(d_range*starvation_factor)
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
        fn.fn.logprint("Stage " + n_stage + ": Crisis! ", False)
        fn.fn.logprint(str(n_survivors)+" individuals survived.")

    # Private methods:

    def __recombine(self, r_rate):
        """Recombine between the two chromosomes of each individual
        in the population."""
        chr1 = np.arange(self.chrlen)
        chr2 = chr1 + self.chrlen
        for n in range(len(self.genomes)):
            g = self.genomes[n]
            r_sites = sample(range(self.chrlen),int(self.chrlen*r_rate))
            r_sites.sort()
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
            pop.genomes = pop.genomes[:-1]
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
            "gen_map":population.genmap,
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
            "starvation_factor":np.copy(array4),
            # Per-age data:
            "age_distribution":np.copy(array1),
            "death_mean":np.copy(array1),
            "death_sd":np.copy(array1),
            "repr_mean":np.copy(array1),
            "repr_sd":np.copy(array1),
            "fitness":np.copy(array1),
            # Genotype data:
            "density_surv":np.copy(array2),
            "density_repr":np.copy(array2),
            "n1":np.zeros([m,population.chrlen]),
            "s1":np.zeros([m,population.chrlen-window_size+1]),
            # Simple per-snapshot data:
            "entropy":np.copy(array3),
            "junk_death":np.copy(array3),
            "junk_repr":np.copy(array3),
            "junk_fitness":np.copy(array3)
            }

    def quick_update(self, n_stage, pop_size, resources, starv_factor):
        """Record only population size, resource and starvation data."""
        self.record["population_size"][n_stage] = pop_size
        self.record["resources"][n_stage] = resources
        self.record["starvation_factor"][n_stage] = starv_factor

    def update_agestats(self, population, n_snap):
        """Record detailed per-age statistics of population at 
        current snapshot stage."""
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
        for age in range(max(p.ages)):
            pop = p.genomes[p.ages==age]
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
            else: # If no individuals of that age, make everything 0.
                death_mean[age] = 0
                death_sd[age] = 0
                if age >= p.maturity:
                    repr_mean[age] = 0
                    repr_sd[age] = 0
        # Average densities over whole genomes
        density_surv /= float(p.N)
        density_repr /= float(p.N)
        # Update record
        agedist = np.bincount(p.ages, minlength = p.maxls)/float(p.N)
        self.record["age_distribution"][n_snap] = agedist
        self.record["death_mean"][n_snap] = death_mean
        self.record["death_sd"][n_snap] = death_sd
        self.record["repr_mean"][n_snap] = repr_mean
        self.record["repr_sd"][n_snap] = repr_sd
        self.record["density_surv"][n_snap] = density_surv
        self.record["density_repr"][n_snap] = density_repr

    def update_invstats(self, population, n_snap):
        """Record detailed cross-population statistics at current 
        snapshot stage."""
        p = population
        # Frequency of 1's at each position on chromosome:
        # Average at each position, then between chromosomes
        n1s = np.sum(p.genomes, 0)/float(p.N)
        n1 = (n1s[np.arange(p.chrlen)]+n1s[np.arange(p.chrlen)+p.chrlen])
        n1 /= 2
        # Shannon-Weaver entropy over entire genome genomes
        p1 = np.sum(p.genomes)/float(p.genomes.size)
        entropy = st.entropy(np.array([1-p1, p1]))
        # Junk stats calculated from neutral locus
        neut_locus = np.nonzero(p.genmap==201)[0][0]
        neut_pos = np.arange(neut_locus*p.nbase, (neut_locus+1)*p.nbase)
        neut_pop = p.genomes[:,np.append(neut_pos, neut_pos+p.chrlen)]
        neut_gen = np.sum(neut_pop, axis=1)
        junk_death = np.mean(self.record["d_range"][neut_gen])
        junk_repr = np.mean(self.record["r_range"][neut_gen]) # Junk SDs?
        # Append record object
        self.record["n1"][n_snap] = n1
        self.record["entropy"][n_snap] = entropy
        self.record["junk_death"][n_snap] = junk_death
        self.record["junk_repr"][n_snap] = junk_repr

    def update(self, population, resources, starv_factor, stage, n_snap):
        """Record detailed population data at current snapshot stage."""
        self.quick_update(stage, population.N, resources, starv_factor)
        self.update_agestats(population, n_snap)
        self.update_invstats(population, n_snap)

    def final_update(self, n_run, window):
        """After run completion, compute fitness and n1 std."""
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
