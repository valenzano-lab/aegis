########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Record                                                      #
# Description: Infodict class containing information recorded during   #
#   the course of a simulation, as well as methods for computing       #
#   more advanced statistics from that information.                    #
########################################################################
# TODO: add generation and parental-age recording
# TODO: convert some output into pandas data frames for easier plotting

## PACKAGE IMPORT ##
import numpy as np
import scipy.stats as st
from .Config import Infodict
from .Population import Population, Outpop
import copy

## CLASS ##

class Record(Infodict):
    """Infodict object for recording and processing simulation data."""

    # INITIALISATION
    def __init__(self, conf):
        """Import run parameters from a Config object and prepare for
        further data input."""
        # Inherit run parameters from Config object
        self.__valdict__ = conf.__valdict__.copy()
        self.__infdict__ = conf.__infdict__.copy()
        #for k in self.keys(): # Put numeric data in arrays #!: Why?
        #    if type(self[k]) in [int, float]:
        #        self[k] = np.array([self[k]]) #! Check required data is there
        # Clearer name for R and V parameters
        self.put("res_regen_constant", self.get_value("R"), self.get_info("R"))
        self.put("res_regen_prop", self.get_value("V"), self.get_info("V"))
        # Basic run info
        self.put("dieoff", np.array(False),
                "Boolean specifying whether run ended in extinction.")
        self.put("prev_failed", np.array(0),
                "Number of run repeats before this one that ended in dieoff.")
        # Arrays for per-stage data entry
        a0 = np.zeros(self["number_of_stages"])
        a1 = np.zeros([self["number_of_stages"],self["max_ls"]])
        self.put("population_size", np.copy(a0),
                "Size of population at each stage.")
        self.put("resources", np.copy(a0),
                "Number of resources available at each stage.")
        self.put("surv_penf", np.copy(a0),
                "Survival penalty due to starvation at each stage.")
        self.put("repr_penf", np.copy(a0),
                "Reproduction penalty due to starvation at each stage.")
        self.put("age_distribution", np.copy(a1),
                "Proportion of population in each age group at each stage.")
        # Space for saving population state for each snapshot
        self.put("snapshot_pops", [0]*self["number_of_snapshots"],
                "Complete population state at each snapshot stage.")
        # Initialise arrays for data entry
        def putzero(name, infstr): self.put(name, 0, infstr)
        # Final population
        putzero("final_pop", "Final population state at end of simulation.")
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
        putzero("mean_repr", "True mean reproductive probability at each age,\
                including juvenile ages, adjusted for sexuality.")
        putzero("junk_repr", "Junk mean reproductive probability at each age,\
                including juvenile ages, adjusted for sexuality.")
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

    ##  FINALISATION  ##

    # 1: GENOTYPE DENSITY DISTRIBUTIONS AND STATISTICS

    def compute_locus_density(self):
        """Compute normalised distributions of sum genotypes for each locus in
        the genome at each snapshot, for survival, reproduction, neutral and
        all loci."""
        l,m,ns = self["max_ls"], self["maturity"], self["n_states"]
        loci_all = np.array([p.toPop().sorted_loci() \
                for p in self["snapshot_pops"]])
        loci = {"s":np.array([L[:,:l] for L in loci_all]),
                "r":np.array([L[:,l:(2*l-m)] for L in loci_all]),
                "n":np.array([L[:,(2*l-m):] for L in loci_all]), "a":loci_all}
        def density(x):
            bins = np.bincount(x,minlength=ns)
            return bins/float(sum(bins))
        density_per_locus = {}
        for k in ["s","r","n","a"]: # Survival, reproductive, neutral, all
            # loci[k]: dim0 = snapshot, dim1 = genotype, dim2 = locus
# NOTE following two lines are obsolete
#            for n in xrange(len(loci)):
#                d = np.apply_along_axis(density,0,loci[k][n])
            out = np.array([np.apply_along_axis(density,0,x) for x in loci[k]])
            density_per_locus[k] = out.transpose(0,2,1)
            # now: dim0 = snapshot, dim1 = locus, dim2 = genotype
        self["density_per_locus"] = density_per_locus

    def compute_total_density(self):
        """Compute overall genotype sum distributions at each snapshot from
        pre-computed per-locus distributions."""
        density = {}
        for k in ["s","r","n","a"]: # Survival, reproductive, neutral, all
            collapsed = np.sum(self["density_per_locus"][k], 1).T
            # collapsed: dim0 = genotype, dim1 = snapshot
            density[k] = (collapsed/np.sum(collapsed, 0)).T
            # density[k]:dim0 = snapshot, dim1 = genotype
        self["density"] = density

    def compute_genotype_mean_var(self):
        """Compute the mean and variance in genotype sums at each locus
        and snapshot."""
        ss,gt = self["snapshot_stages"],np.arange(self["n_states"])
        mean_gt_dict, var_gt_dict = {}, {}
        for k in ["s","r","n","a"]: # Survival, reproductive, neutral, all
            dl = self["density_per_locus"][k] #[snapshot,locus,genotype]
            mean_gt = np.sum(dl * gt, 2) # [snapshot, locus]
            # Get difference between each potential genotype and the mean at
            # each snapshot/locus, then compute variance
            gt_diff = np.tile(gt, [len(ss),dl.shape[1],1]) - \
                    np.repeat(mean_gt[:,:,np.newaxis], len(gt), axis=2)
            var_gt = np.sum(dl * (gt_diff**2), 2) # [snapshot, locus]
            mean_gt_dict[k],var_gt_dict[k] = mean_gt,var_gt
        self["mean_gt"] = mean_gt_dict
        self["var_gt"] = var_gt_dict

    # SURVIVAL/REPRODUCTION PROBABILITIES, FITNESS, AND REPRODUCTIVE VALUE

    def compute_surv_repr_probabilities_true(self):
        """Compute true mean and variance in survival and reproduction
        probability at each age and snapshot from the corresponding genotype
        distributions at survival and reproduction loci."""
        # Get mean and variance in genotype sums
        mean_gt, var_gt = self["mean_gt"], self["var_gt"]
        keys, functions = ["surv","repr"], [self.p_surv, self.p_repr]
        prob_mean, prob_var = {}, {}
        for n in xrange(2):
            k,f = keys[n], functions[n]
            prob_mean[k] = f(mean_gt[k[0]])
            prob_var[k] = var_gt[k[0]]*self[k+"_step"]
        self["prob_mean"] = prob_mean
        self["prob_var"] = prob_var

    def compute_surv_repr_probabilities_junk(self):
        """Compute junk mean and variance in survival and reproduction
        probability at each age and snapshot from the average genotype
        distribution across all neutral loci."""
        # Get mean and variance in genotype sums
        mean_gt, var_gt = self["mean_gt"], self["var_gt"]
        keys, functions = ["surv","repr"], [self.p_surv, self.p_repr]
        junk_mean, junk_var = {}, {}
        for n in xrange(2):
            k,f = keys[n], functions[n]
            junk_mean[k] = f(mean_gt["n"])
            junk_var[k] = var_gt["n"]*self[k+"_step"]
        self["junk_mean"] = junk_mean
        self["junk_var"] = junk_var

    def compute_cmv_surv(self):
        """Compute true and junk cumulative survival probabilities at each age
        and snapshot from the corresponding survival probability arrays."""
        l = self["max_ls"]
        init_surv = np.tile(1,len(self["prob_mean"]["surv"]))
        # (P(survival from age 0 to age 0) = 1)
        cmv_surv = np.ones(self["prob_mean"]["surv"].shape)
        cmv_surv[:,1:] = np.cumprod(self["prob_mean"]["surv"],1)[:,:-1]
        #! Ideally should calculate this separately for each neutral locus,
        #! rather than taking the average
        junk_cmv_surv = \
            np.mean(self["junk_mean"]["surv"],1)[:,np.newaxis]**np.arange(l)
        self["cmv_surv"] = cmv_surv
        self["junk_cmv_surv"] = junk_cmv_surv

    def compute_mean_repr(self):
        """Fit mean reproduction probabilities to shape of survival
        probabilities, for downstream computation of fitness and
        reproductive value."""
        sex = self["repr_mode"] in ["sexual", "assort_only"]
        # True values
        mean_repr = np.zeros(self["prob_mean"]["surv"].shape)
        mean_repr[:,self["maturity"]:] = self["prob_mean"]["repr"]
        mean_repr /= 2.0 if sex else 1.0
        self["mean_repr"] = mean_repr
        # Junk values
        q = np.mean(self["junk_mean"]["repr"], 1)
        junk_repr = np.tile(q[:,np.newaxis], [1,self["max_ls"]])
        junk_repr[:,:self["maturity"]] = 0
        junk_repr /= 2.0 if sex else 1.0 #! TODO: Check this
        self["junk_repr"] = junk_repr

    def compute_fitness(self):
        """Compute true and junk per-age and total fitness for each
        snapshot."""
        # Per-age fitness contribution = P(survival to x) x P(repr at x)
        f = self["cmv_surv"] * self["mean_repr"]
        junk_f = self["junk_cmv_surv"] * self["junk_repr"]
        # Total fitness = sum over all per-age contributions
        fitness, junk_fitness = np.sum(f, 1), np.sum(junk_f, 1)
        self["fitness_term"] = f
        self["junk_fitness_term"] = junk_f
        self["fitness"] = fitness
        self["junk_fitness"] = junk_fitness

    def compute_reproductive_value(self):
        """Compute expected future offspring at each snapshot and age; if the
        population size is stable, this is equivalent to the reproductive value
        of each age cohort."""
        def cumsum_rev(a): return np.fliplr(np.cumsum(np.fliplr(a),1))
        # E(future offspring at age x| survival to age x)
        repr_value = cumsum_rev(self["fitness_term"])/self["cmv_surv"]
        junk_repr_value = cumsum_rev(self["junk_fitness_term"])/\
                self["junk_cmv_surv"]
        self["repr_value"] = repr_value
        self["junk_repr_value"] = junk_repr_value

    # MEAN AND VARIANCE IN BIT VALUE

    def compute_bits(self):
        """Compute the mean and variance bit value at each position along the
        chromosome, sorted according to genmap_argsort."""
        """During finalisation, compute the distribution of 1s and 0s at each
        position on the chromosome (sorted by genome map), along with
        associated statistics."""
        l,m,b = self["max_ls"], self["maturity"], self["n_base"]
        # Reshape genomes to stack chromosomes
        # [snapshot, individual, bit]
        stacked_chrs = [p.toPop().genomes.reshape(p.N*2,p.chr_len) \
                for p in self["snapshot_pops"]]
        # Compute order of bits in genome map
        order = np.ndarray.flatten(
            np.array([p.toPop().genmap_argsort*b + c for c in xrange(b)]),
            order="F") # Using last population; genmaps should all be same
        # Average across individuals and sort [snapshot, bit]
        n1 = np.array([np.mean(sc, axis=0)[order] for sc in stacked_chrs])
        n1_var = np.array([np.var(sc, axis=0)[order] for sc in stacked_chrs])
        # Set record entries
        self["n1"] = n1
        self["n1_var"] = n1_var

    # ENTROPY IN GENOTYPES AND BITS

    def compute_entropies(self):
        """Compute the Shannon entropy in genotype and bit values across
        at each snapshot."""
        # Genotypic entropy for each set of loci
        entropy_gt = {}
        for k in ["s","r","n","a"]: # Survival, reproductive, neutral, all
            # NOTE transposed
            d = self["density"][k].T
            entropy_gt[k] = np.apply_along_axis(st.entropy, 0, d)
        self["entropy_gt"] = entropy_gt
        # Bit entropy
        n1_total = np.mean(self["n1"], 1)
        bit_distr = np.vstack((n1_total,1-n1_total))
        entropy_bits = np.apply_along_axis(st.entropy, 0, bit_distr)
        self["entropy_bits"] = entropy_bits
        #! TODO: Also separate bit entropy by type of locus (s,r,n,a)

    # ACTUAL DEATH RATES

    def compute_actual_death(self):
        """Compute actual death rate for each age at each stage."""
        N_age = self["age_distribution"] *\
                self["population_size"][:,None]
        dividend = N_age[1:, 1:]
        divisor = np.copy(N_age[:-1, :-1])
        divisor[divisor == 0] = np.nan # flag division by zero
        self["actual_death_rate"] = 1 - dividend / divisor

    # SLIDING WINDOWS

    def get_window(self, key, wsize):
        """Obtain sliding windows from a record entry."""
        x = self[key]
        d,s = x.ndim-1, len(x.strides)-1
        w = np.min([wsize, x.shape[d] + 1]) # Maximum window size
        a_shape = x.shape[:d] + (x.shape[d] - w + 1, w)
        a_strd = x.strides + (x.strides[s],)
        #print a_shape, a_strd
        return np.lib.stride_tricks.as_strided(x, a_shape, a_strd)

    def compute_windows(self):
        dim = {"population_size":1, "resources":1, "n1":2}
        for s in ["population_size","resources","n1"]:
            w = self.get_window(s, self["windows"][s])
            self[s + "_window_mean"] = np.mean(w, dim[s])
            self[s + "_window_var"] = np.var(w, dim[s])

    # OVERALL

    def finalise(self):
        """Calculate additional stats from recorded data of a completed run."""
        # Subset age distributions to snapshot stages for plotting
        # TODO test this in record test for finalise
        self.put("snapshot_age_distribution",
                self["age_distribution"][self["snapshot_stages"]],
                "Distribution of ages in the population at each snapshot stage."
                )
        # Genotype distributions and statistics
        self.compute_locus_density()
        self.compute_total_density()
        self.compute_genotype_mean_var()
        # Survival/reproduction probabilities
        self.compute_surv_repr_probabilities_true()
        self.compute_surv_repr_probabilities_junk()
        self.compute_cmv_surv()
        self.compute_mean_repr()
        self.compute_fitness()
        self.compute_reproductive_value()
        # Other values
        self.compute_bits()
        self.compute_entropies()
        self.compute_actual_death()
        self.compute_windows()
        # Remove snapshot pops as appropriate
        if self["output_mode"] > 0:
            self["final_pop"] = self["snapshot_pops"][-1]
        if self["output_mode"] < 2:
            self["snapshot_pops"] = 0

    # __startpop__ method

    def __startpop__(self, pop_number):
            if pop_number < 0 and isinstance(self["final_pop"], Population):
                msg = "Setting seed from final population in Record."
                pop = self["final_pop"]
            elif pop_number < 0:
                msg = "Failed to set seed from final population in Record; {}"
                msg = msg.format("(no such population).")
                pop = ValueError
            elif pop_number >= self["number_of_snapshots"]:
                msg = "Seed number ({0}) greater than highest snapshot ({1})."
                msg = msg.format(pop_number, self["number_of_snapshots"]-1)
                pop = ValueError
            else:
                msg = "Setting seed from specified snapshot population ({})."
                msg = msg.format(pop_number)
                pop = self["snapshot_pops"][pop_number]
            return (pop, msg)

    # copy method

    def copy(self): return copy.deepcopy(self)
