########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Core                                                         #
# Classes: Record                                                      #
# Description: Dictionary class containing information recorded during #
#   the course of a simulation, as well as methods for computing       #
#   more advanced statistics from that information.                    #
########################################################################

## PACKAGE IMPORT ##
import numpy as np
import scipy.stats as st
from .functions import fivenum, deep_eq, deep_key
from .Config import Config
from .Population import Population
import copy

## CLASS ##

class Record(dict):
    """Dictionary object for recording and processing simulation data."""

    # INITIALISATION
    def __init__(self, conf):
        """Import run parameters from a Config object and prepare for
        further data input."""
        # Inherit run parameters from Config object
        for k in conf.keys():
            if k in ["res_function", "stv_function"]: # Can't save function objects
                self[k] = 0
            else:
                self[k] = conf[k]
        # Basic run info
        self["dieoff"] = np.array(False)
        self["prev_failed"] = np.array(0)
        self["finalised"] = False
        # Arrays for per-stage data entry
        n = self["n_stages"] if not self["auto"] else self["max_stages"]
        ns,ml,mt = self["n_snapshots"], self["max_ls"], self["maturity"]
        for k in ["population_size", "resources", "surv_penf", "repr_penf"]:
            self[k] = np.zeros(n)
        self["age_distribution"] = np.zeros([n, ml])
        for k in ["generation_dist", "gentime_dist"]:
            self[k] = np.zeros([n,5]) # Five-number summaries
        # Space for saving population state for each snapshot
        self["snapshot_pops"] = [0]*ns
        # Basic properties of snapshot populations
        for l in ["age", "gentime"]:
            self["snapshot_{}_distribution".format(l)] = np.zeros([ns,ml])
        self["snapshot_generation_distribution"] = np.zeros(
                [ns, np.ceil(conf["object_max_age"]/float(mt)).astype(int)+1])
        if conf["auto"]:
            self["snapshot_stages"] = np.zeros([ns])

    # CALCULATING PROBABILITIES
    def p_scale(self, bound):
        """Compute the scaling factor relating genotype sums to
        probabilities for a given set of probability bounds."""
        p_min,p_max = np.array(bound).astype(float)
        return (p_max - p_min)/(2*self["n_base"])
    def p_shift(self, bound):
        """Compute the shifting factor relating genotype sums to
        probabilities for a given set of probability bounds."""
        p_min,p_max = np.array(bound).astype(float)
        return p_min
    def p_calc(self, gt, bound):
        """Derive a probability array from a genotype array and list of
        max/min values."""
        # Check for invalid genotype values
        minval, maxval = np.min(gt), np.max(gt)
        if minval < 0:
            raise ValueError("Invalid genotype value: {}".format(minval))
        if maxval > 2*self["n_base"]:
            raise ValueError("Invalid genotype value: {}".format(maxval))
        return self.p_scale(bound)*gt + self.p_shift(bound) # Y=aX+b
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
        self["generation_dist"][n_stage] = fivenum(population.generations)
        self["gentime_dist"][n_stage] = fivenum(population.gentimes)
        if n_snap >= 0:
            self["snapshot_pops"][n_snap] = population.clone()

    ##  FINALISATION  ##

    # 0: BASIC PROPERTIES (AGES, GENERATIONS, ETC)

    def compute_snapshot_properties(self):
        """Compute basic properties (ages, gentimes, etc) of snapshot
        populations during finalisation."""
        #n = self["n_stages"] if not self["auto"] else self["max_stages"]
        n = self["object_max_age"]
        g = np.ceil(n/float(self["maturity"])).astype(int)+1
        for s in xrange(self["n_snapshots"]):
            p = self["snapshot_pops"][s]
            minlen = {"age":p.max_ls, "gentime":p.max_ls, "generation":g}
            for k in ["age", "gentime", "generation"]:
                key = "snapshot_{}_distribution".format(k)
                newval = np.bincount(getattr(p, "{}s".format(k)),
                        minlength=minlen[k])/float(p.N)
                self[key][s] = newval

    def compute_locus_density(self):
        """Compute normalised distributions of sum genotypes for each locus in
        the genome at each snapshot, for survival, reproduction, neutral and
        all loci."""
        l,m,ns = self["max_ls"], self["maturity"], self["n_states"]
        loci_all = np.array([p.sorted_loci() \
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
        ss = self["snapshot_generations" if self["auto"] else "snapshot_stages"]
        gt = np.arange(self["n_states"])
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
        keys = ["surv","repr"]
        prob_mean, prob_var = {}, {}
        for n in xrange(2):
            k = keys[n]
            bound = self["{}_bound".format(k)]
            scale, shift = self.p_scale(bound), self.p_shift(bound)
            prob_mean[k] = scale*mean_gt[k[0]] + shift # E(Y) = aE(X)+b
            prob_var[k] = (scale**2)*var_gt[k[0]] # Var(Y) = (a^2)Var(X)
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
        junk_repr /= 2.0 if sex else 1.0
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
        stacked_chrs = [p.genomes.reshape(p.N*2,p.chr_len) \
                for p in self["snapshot_pops"]]
        # Compute order of bits in genome map
        order = np.ndarray.flatten(
            np.array([p.genmap_argsort*b + c for c in xrange(b)]),
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
            d = self["density"][k].T
            entropy_gt[k] = np.apply_along_axis(st.entropy, 0, d)
        self["entropy_gt"] = entropy_gt
        # Bit entropy
        n1_total = np.mean(self["n1"], 1)
        bit_distr = np.vstack((n1_total,1-n1_total))
        entropy_bits = np.apply_along_axis(st.entropy, 0, bit_distr)
        self["entropy_bits"] = entropy_bits

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
        # If dieoff, truncate data to last snapshot pop
        if self["dieoff"]:
            pops = np.array(self["snapshot_pops"])
            pops = pops[np.nonzero(pops)]
            self["snapshot_pops"] = list(pops)
            self["n_snapshots"] = len(self["snapshot_pops"])
            if self["auto"]:
                self["snapshot_generations"] = self["snapshot_generations"][\
                    :self["n_snapshots"]]
            self["snapshot_stages"] = self["snapshot_stages"][:self["n_snapshots"]]
        # Compute basic properties of snapshot pops
        self.compute_snapshot_properties()
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
        self["finalised"] = True

    # copy method

    def copy(self):
        sc_prng = self["prng"]
        self_copy = copy.deepcopy(self)
        self_copy["prng"] = sc_prng
        self_copy["params"]["prng"] = sc_prng
        return self_copy

    # COMPARISON (same as Config)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return deep_eq(self, other, True)
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, self.__class__): return not self.__eq__(other)
        return NotImplemented
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
