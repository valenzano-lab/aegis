########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Record                                                      #
# Description: Infodict class containing information recorded during   #
#   the course of a simulation, as well as methods for computing       #
#   more advanced statistics from that information.                    #
########################################################################

## PACKAGE IMPORT ##
import numpy as np
from .Config import Infodict

## CLASS ##

class Record(Infodict):
    """Infodict object for recording and processing simulation data."""

    # INITIALISATION
    def __init__(self, conf):
        """Import run parameters from a Config object and prepare for
        further data input."""
        # Inherit run parameters from Config object
        self.__valdict__ = conf.__valdict__
        self.__infdict__ = conf.__infdict__
        for k in self.keys(): # Put numeric data in arrays #!: Why?
            if type(self[k]) in [int, float]: 
                self[k] = np.array([self[k]]) #! Check required data is there
        # Clearer name for R and V parameters
        self.put("res_regen_constant", self.get_value("R"), self.get_info("R"))
        self.put("res_regen_prop", self.get_value("V"), self.get_info("V"))
        # Basic run info
        self.put("dieoff", np.array(False), 
                "Boolean specifying whether run ended in extinction.")
        self.put("prev_failed", np.array(0),
                "Number of run repeats before this one that ended in dieoff.")
        self.put("percent_dieoff", np.array(0),
                "Number of all run repeats (including this one) that ended\
                        in dieoff.")
        # Arrays for per-stage data entry
        a0 = np.zeros(self["number_of_stages"])
        a1 = np.zeros([self["number_of_stages"],self["max_ls"]])
        self.put("population_size", a0,
                "Size of population at each stage.")
        self.put("resources", a0,
                "Number of resources available at each stage.")
        self.put("surv_penf", a0,
                "Survival penalty due to starvation at each stage.")
        self.put("repr_penf", a0,
                "Reproduction penalty due to starvation at each stage.")
        self.put("age_distribution", a1,
                "Proportion of population in each age group at each stage.")
        # Space for saving population state for each snapshot
        self.put("snapshot_pops", [0]*self["number_of_snapshots"],
                "Complete population state at each snapshot stage.")
        # Initialise arrays for data entry
        def putzero(name, infstr): self.put(name, 0, infstr)
        # Genotype sum statistics (density and average)
        putzero("density_per_locus",
                "Density distributions of genotype sums (from 0 to maximum)\
                        at each locus in genome map at each snapshot,\
                        for survival, reproduction, neutral and all loci.")
        putzero("density",
                "Density distribution of genotype sums (from 0 to maximum)\
                        over all loci of each type, for survival,\
                        reproduction, neutral and all loci.")
        putzero("mean_gt", "Mean genotype sum value at each locus at each\
                snapshot, for survival, reproduction, neutral and all loci.")
        putzero("var_gt", "Variance in genotype sum value at each locus\
                at each snapshot, for survival, reproduction, neutral and\
                all loci.")
        putzero("entropy_gt", "Shannon entropy measurement of genotype\
                sum diversity over all loci at each snapshot, for survival,\
                reproduction, neutral and all loci.")
        # Survival and reproduction
        putzero("cmv_surv", "Cumulative survival probability from age\
                0 to age n at each snapshot, based on corresponding\
                survival loci.")
        putzero("junk_cmv_surv", "Cumulative survival probability from \
                age 0 to age n at each snapshot, based on average over \
                neutral loci.")
        putzero("prob_mean", "Mean probability of survival/reproduction\
                at each age at each snapshot, \
                based on corresponding locus genotypes")
        putzero("prob_var", "Variance in  probability of survival/\
                reproduction at each age at each snapshot, \
                based on corresponding locus genotypes")
        putzero("junk_mean", "Mean probability of survival/reproduction\
                at each age at each snapshot, \
                based on average over neutral loci")
        putzero("junk_var", "Variance in  probability of survival/\
                reproduction at each age at each snapshot, \
                based on average over neutral loci")
        putzero("fitness_term", "Expected offspring at each age at each\
                snapshot, equal to cumulative survival to that age * \
                probability of reproduction at that age, based on \
                corresponding survival/reproduction loci.")
        putzero("junk_fitness_term", "Expected offspring at each age at \
                each snapshot, equal to cumulative survival to that age * \
                probability of reproduction at that age, based on \
                average over neutral loci.")
        putzero("fitness", "Mean true \
                genotypic fitness for population at each snapshot, equal to \
                the sum of true fitness terms over all ages.")
        putzero("junk_fitness", "Mean junk \
                genotypic fitness for population at each snapshot, equal to \
                the sum of junk fitness terms over all ages.")
        putzero("repr_value", "Mean reproductive value of individuals \
                at each age at each snapshot, assuming the population is \
                stable (equivalent to expected future offspring).")
        putzero("junk_repr_value", "Mean junk reproductive value of \
                individuals at each age in each snapshot, assuming the \
                population is stable (computed from mean over neutral loci.")
        # Per-bit statistics, actual death
        putzero("n1",
                "Mean value at each bit position in genome at each snapshot,\
                ordered by age-value of corresponding locus.")
        putzero("n1_var",
                "Variance in value at each bit position in genome at each \
                snapshot, ordered by age-value of corresponding locus.")
        putzero("entropy_bits",
                "Shannon-entropy measurement of bit-value diversity over \
                all bit positions at each snapshot.")
        putzero("actual_death_rate",
                "Actual death rate for each age cohort at each stage.")
        # Sliding windows
        putzero("population_size_window_mean", "Sliding-window mean of\
                population sizes over stages of simulation.")
        putzero("population_size_window_var", "Sliding-window variance in\
                population sizes over stages of simulation.")
        putzero("resources_window_mean", "Sliding-window mean of\
                resource levels over stages of simulation.")
        putzero("resources_window_var", "Sliding-window variance in\
                resource levels over stages of simulation.")
        putzero("n1_window_mean", "Sliding-window mean of\
                average bit value over age-ordered bits in the chromosome")
        putzero("n1_window_var", "Sliding-window variance in\
                average bit value over age-ordered bits in the chromosome")

    # CALCULATING PROBABILITIES
    def p_calc(self, gt, bound):
        """Derive a probability array from a genotype array and list of 
        max/min values."""
        minval, maxval, limit = np.min(gt), np.max(gt), 2*self["n_base"]
        if minval < 0:
            raise ValueError("Invalid genotype value: {}".format(minval))
        if maxval > limit:
            raise ValueError("Invalid genotype value: {}".format(maxval))
        p_min,p_max = np.array(bound).astype(float)
        return p_min + (p_max - p_min)*gt/limit
    def p_surv(self, gt):
        """Derive an array of survival probabilities from a genotype array."""
        return self.p_calc(gt, self["surv_bound"])
    def p_repr(self, gt):
        """Derive an array of reproduction probabilities from a genotype array."""
        return self.p_calc(gt, self["repr_bound"])

    # PER-STAGE RECORDING
    def update(self, population, resources, surv_penf, repr_penf, n_stage, 
            n_snap=-1):
        """Record per-stage data (population size, age distribution, resources,
        and survival penalties), plus, if on a snapshot stage, the population
        as a whole."""
        self["population_size"][n_stage] = population.N
        self["resources"][n_stage] = resources
        self["surv_penf"][n_stage] = surv_penf
        self["repr_penf"][n_stage] = repr_penf
        self["age_distribution"][n_stage] = np.bincount(population.ages,
                minlength = population.max_ls)/float(population.N)
        if n_snap >= 0:
            self["snapshot_pops"][n_snap] = Outpop(population)
        #! Consider snapshot_pops recording: write to a tempdir instead?

    # FINALISATION
    def compute_densities(self):
        """During finalisation, compute per-age and overall genotype density
        distributions for different types of loci at each snapshot, along with
        mean, variance and entropy in genotype sum."""
        l,m,b = self["max_ls"], self["maturity"], self["n_base"]
        ad,ss = self["age_distribution"], self["snapshot_stages"]
        ns,gt = self["n_states"], np.arange(self["n_states"])
        # 0: Auxiliary functions:
        def density_by_locus(loci):
            """Return the normalised distributions of sum genotypes for each 
            column in a locus array."""
            def density(x):
                bins = np.bincount(x,minlength=gt)
                return bins/float(sum(bins))
            # dim0 = snapshot, dim1 = genotype, dim2 = age
            out = np.array([np.apply_along_axis(density,0,x) for x in loci])
            # dim0 = genotype, dim1 = snapshot, dim2 = locus
            return out.transpose(1,0,2)
        def total_density(locus_densities):
            """Compute an overall density distribution from an array of
            per-locus distributions."""
            collapsed = np.sum(locus_densities, 2)
            # dim0 = genotype, dim1 = snapshot
            return collapsed/np.sum(collapsed, 0)
        def get_mean_var_gt(locus_densities):
            """Get the per-locus mean and variance of a genotype distribution
            from an array of densities."""
            ad_tr = locus_densities.transpose(1,2,0) #[snapshot,locus,genotype]
            # Mean and variance of gt distribution
            mean_gt = np.sum(ad_tr * gt, 2) # [snapshot, locus]
            # Get difference between each potential genotype and the mean at
            # each snapshot/locus
            gt_diff = np.tile(gt, [len(ss),ad_tr.shape[1],1]) - \
                    np.repeat(mean_gt[:,:,np.newaxis], len(gt), axis=2)
            var_gt = np.sum(ad_tr * (gt_diff**2), 2)
            return [mean_gt, var_gt]
        loci = np.array([p.toPop().sorted_loci() \
                for p in self["snapshot_pops"]])
        loci_by_type = {"s":np.array([L[:,:l] for L in loci]),
                "r":np.array([L[:,l:(2*l-m)] for L in loci]),
                "n":np.array([L[:,(2*l-m):] for L in loci]), "a":loci}
        # Dimensions will differ depending on population size at each snapshot,
        # so can't collapse into a single array yet.
        density_per_locus,density,mean_gt,var_gt,entropy_gt = {},{},{},{},{}
        for k in ["s","r","n","a"]: # Survival, reproductive, neutral, all
            density_per_locus[k] = density_by_locus(loci_by_type[k])
            density[k] = total_density(density_per_locus[k])
            mean_gt[k], var_gt[k] = get_mean_var_gt(density_per_locus[k])
            entropy_gt[k] = np.apply_along_axis(st.entropy, 0, density[k])
        # Set record entries
        self["density_per_locus"] = density_per_locus
        self["density"] = density
        self["mean_gt"] = mean_gt
        self["var_gt"] = var_gt
        self["entropy_gt"] = entropy_gt

    def compute_probabilities(self):
        """During finalisation, compute mean and variance in survival and 
        reproduction probability at each snapshot, along with the resulting
        fitness and reproductive value."""
        l,m,b = self["max_ls"], self["maturity"], self["n_base"]
        # Simple surv/repr probabilities
        mean_gt, var_gt = self["mean_gt"], self["var_gt"]
        prob_mean, prob_var, junk_mean, junk_var = {},{},{},{}
        keys, fns = ["surv","repr"],[self.p_surv,self.p_repr]
        for n in xrange(2):
            k,f = keys[n], fns[n]
            prob_mean[k] = f(mean_gt[k[0]])
            prob_var[k] = var_gt[k[0]]*self[k+"_step"]
            junk_mean[k] = f(mean_gt["n"])
            junk_var[k] = var_gt["n"]*self[k+"_step"]
        # Cumulative survival probabilities: P(survival from birth to age x)
        init_surv = np.tile(1,len(prob_mean["surv"]))
        # (P(survival from age 0 to age 0) = 1)
        cmv_surv = np.ones(prob_mean["surv"].shape)
        cmv_surv[:,1:] = np.cumprod(prob_mean["surv"],1)[:,:-1]
        junk_cmv_surv = np.mean(junk_mean["surv"],1)[:,np.newaxis]**np.arange(l)
        #! Ideally should calculate this separately for each neutral locus,
        #! rather than taking the average
        # Fit reproduction probs to shape of survival probs
        mean_repr = np.zeros(prob_mean["surv"].shape)
        mean_repr[:,m:] = prob_mean["repr"]
        mean_repr /= 2.0 if self["sexual"] else 1.0
        junk_repr = np.zeros(prob_mean["surv"].shape)
        junk_repr[:,m:] = np.mean(junk_mean["repr"],1)[:,np.newaxis]
        junk_repr /= 2.0 if self["sexual"] else 1.0
        # Per-age fitness contribution = P(survival to x) x P(reproduction at x)
        f = cmv_surv * mean_repr
        fitness = np.sum(f, 1)
        junk_f = junk_cmv_surv * junk_repr
        junk_fitness = np.sum(junk_f, 1)
        # Reproductive value = E(future offspring at age x| survival to age x)
        def cumsum_rev(a): return np.fliplr(np.cumsum(np.fliplr(a),1))
        repr_value = cumsum_rev(f)/cmv_surv
        junk_repr_value = cumsum_rev(junk_f)/junk_cmv_surv
        # Set record entries
        self["cmv_surv"] = cmv_surv
        self["junk_cmv_surv"] = junk_cmv_surv
        self["prob_mean"] = prob_mean
        self["prob_var"] = prob_var
        self["junk_mean"] = junk_mean
        self["junk_var"] = junk_var
        self["fitness_term"] = f
        self["junk_fitness_term"] = junk_f
        self["fitness"] = fitness
        self["junk_fitness"] = junk_fitness
        self["repr_value"] = repr_value
        self["junk_repr_value"] = junk_repr_value

    def compute_bits(self):
        """During finalisation, compute the distribution of 1s and 0s at each
        position on the chromosome (sorted by genome map), along with
        associated statistics."""
        l,m,b = self["max_ls"], self["maturity"], self["n_base"]
        # Reshape genomes to stack chromosomes
        # [snapshot, individual, bit]
        stacked_chrs = [p.toPop().genomes.reshape(p.N*2,p.chrlen) \
                for p in self["snapshot_pops"]]
        # Dimensions will differ depending on population size at each snapshot,
        # so can't collapse into a single array yet.
        # Compute order of bits in genome map
        order = np.ndarray.flatten(
            np.array([p.toPop().genmap_argsort*b + c for c in xrange(b)]),
            order="F") # Using last population; genmaps should all be same
        # Average across individuals and sort [snapshot, bit]
        n1 = np.array([np.mean(sc, axis=0)[order] for sc in stacked_chrs])
        n1_var = np.array([np.var(sc, axis=0)[order] for sc in stacked_chrs])
        # Compute overall frequency of 1's at each snapshot
        n1_total = np.mean(n1, 1)
        bit_distr = np.vstack((n1_total,1-n1_total))
        entropy_bits = np.apply_along_axis(st.entropy, 0, bit_distr)
        # Set record entries
        self["n1"] = n1
        self["n1_var"] = n1_var
        self["entropy_bits"] = entropy_bits
        #! Add different values for survival, reproduction, neutral, all

    def compute_actual_death(self):
        """Compute actual death rate for each age at each stage."""
        N_age = self["age_distribution"] *\
                self["population_size"][:,None]
        dividend = N_age[1:, 1:]
        divisor = np.copy(N_age[:-1, :-1])
        divisor[divisor == 0] = np.nan # flag division by zero
        self["actual_death_rate"] = 1 - dividend / divisor

    def get_window(self, key, wsize):
        """Obtain sliding windows from a record entry."""
        x = self[key]
        d,s = x.ndim-1, len(x.strides)-1
        w = np.min([wsize, x.shape[d] + 1]) # Maximum window size
        a_shape = x.shape[:d] + (x.shape[d] - w + 1, w)
        a_strd = x.strides + (x.strides[s],)
        return np.lib.stride_tricks.as_strided(x, a_shape, a_strd)

    def compute_windows(self):
        dim = {"population_size":1, "resources":1, "n1":2}
        for s in ["population_size","resources","n1"]:
            w = self.get_window(s, self["windows"][s])
            self[s + "_window_mean"] = np.mean(w, dim[s])
            self[s + "_window_var"] = np.var(w, dim[s])

    def finalise(self):
        """Calculate additional stats from recorded data of a completed run."""
        self.compute_densities()
        self.compute_probabilities()
        self.compute_bits()
        self.compute_actual_death()
        self.compute_windows()
        self["snapshot_pops"] = 0 # TODO: Make this configurable!
