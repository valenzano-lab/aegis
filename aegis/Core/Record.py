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
from .functions import fivenum, deep_eq, deep_key, make_windows
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
            if k in ["res_function", "stv_function", "surv_pen_func", "repr_pen_func"]: # Can't save function objects
                self[k] = 0
            else:
                self[k] = conf[k]
        # Basic run info
        self["dieoff"] = np.array(False)
        self["dieoff_at"] = np.zeros((conf["max_fail"],2)) if conf["auto"] else\
                np.zeros(conf["max_fail"]) # for auto, records stage and generation at dieoff
        self["prev_failed"] = np.array(0)
        self["finalised"] = False
        self["age_dist_truncated"] = False
        # Arrays for per-stage data entry
        n = self["n_stages"] if not self["auto"] else self["max_stages"]
        ns,ml,mt = self["n_snapshots"], self["max_ls"], self["maturity"]
        for k in ["population_size", "resources"]:
            self[k] = np.zeros(n)
        self["starvation_flag"] = np.zeros(n).astype(int)
        self["age_distribution"] = np.zeros([n, ml])
        self["observed_repr_rate"] = np.zeros([n, ml])
#        self["bit_variance"] = np.zeros([n,2])
        for k in ["generation_dist", "gentime_dist"]:
            self[k] = np.zeros([n,5]) # Five-number summaries
        # Space for saving population state for each snapshot
        self["snapshot_pops"] = [0]*ns
        # Basic properties of snapshot populations
        for l in ["age", "gentime"]:
            self["snapshot_{}_distribution".format(l)] = np.zeros([ns,ml])
        self["snapshot_generation_distribution"] = []
        if conf["auto"]:
            self["snapshot_stages"] = np.zeros([ns])
            self["age_dist_stages"] = [[] for i in xrange(ns)]

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
    def update(self, population, resources, starvation_flag, n_stage,
            n_snap=-1, around_snap=-1):
        """Record per-stage data (population size, age distribution, resources,
        and survival penalties), plus, if on a snapshot stage, the population
        as a whole."""
        self["population_size"][n_stage] = population.N
        self["resources"][n_stage] = resources
        self["starvation_flag"][n_stage] = starvation_flag
        if around_snap > -1:
            agehist =  np.bincount(population.ages,minlength = population.max_ls)
            self["age_distribution"][around_snap] = agehist / float(population.N)
            parent_ages = population.gentimes[population.ages==0].flatten()
            agehist[agehist==0] = 1 # avoid division by zero
            self["observed_repr_rate"][around_snap] = np.bincount(parent_ages,\
                    minlength = population.max_ls)/agehist.astype(float)
        self["generation_dist"][n_stage] = fivenum(population.generations)
        # prepare gentimes if assortment takes place
        prep_gentimes = population.gentimes.sum(1)/2 if self["repr_mode"] in\
                ["assort_only","sexual"] else population.gentimes
        self["gentime_dist"][n_stage] = fivenum(prep_gentimes)
        # divide bit variance in two bins: loci before and after maturity
        where_premature = np.zeros(self["chr_len"])
        where_premature[self["maturity"]*self["n_base"]:] = 1
        where_premature[self["max_ls"]*self["n_base"]:(self["max_ls"]+self["maturity"])*\
                self["n_base"]] = 1
        where_premature = where_premature.astype(bool)
        where_mature = np.invert(where_premature)
        where_mature[(2*self["max_ls"]-self["maturity"])*self["n_base"]:] = False
#        self["bit_variance"][n_stage] = np.array([\
#                np.mean(self.compute_bits(population)[1][where_premature]),\
#                np.mean(self.compute_bits(population)[1][where_mature])])
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
                if k == "gentime" and self["repr_mode"] in ["assort_only","sexual"]:
                    # compute rounded mean of two parents
                    attr = getattr(p, "{}s".format(k))
                    newval = np.bincount(attr.sum(1)/2,
                            minlength=minlen[k])/float(p.N)
                else:
                    newval = np.bincount(getattr(p, "{}s".format(k)),
                            minlength=minlen[k])/float(p.N)
                if k == "generation":
                    # save only nonzero values in a 2 column matrix, where
                    # the first column is the generation and the second it's
                    # distribution
                    nonzero = np.nonzero(newval)
                    newval = np.dstack((nonzero, newval[nonzero]))[0]
                    self[key].append(newval)
                else:
                    self[key][s] = newval

    def compute_snapshot_age_dist_avrg(self):
        if not self["age_dist_N"] == "all":
            """Compute snapshot-averaged age distribution."""
            self.truncate_age_dist()
            if self["age_dist_stages"].size > 0:
                self["snapshot_age_distribution_avrg"] = \
                    self["age_distribution"].mean(1)

    def compute_flag_lengths(self, reverse=False):
        """Compute the lengths of starvation periods."""
        stv_flag = self["starvation_flag"]
        if reverse: stv_flag = np.array([1,0])[stv_flag]
        try:
            # check that there is at least on transition
            x1 = np.where(stv_flag==0)[0][0]            # index of first 0
            y1 = np.where(stv_flag==1)[0][0]            # index of first 1
            x2 = np.where(stv_flag[y1:]==0)[0][0] + y1  # index of first 0 after first 1
            y2 = np.where(stv_flag[x1:]==1)[0][0] + x1  # index of first 1 after first 0
            if (x1 < y1 and y1 < x2) or (y1 < x1 and x1 < y2):
                # mark transitions; starvation to no starvation with 1 and vice versa with -1
                flags = stv_flag[:-1] - stv_flag[1:]
                stv2no = np.where(flags==1)[0] + 1
                no2stv = np.where(flags==-1)[0] + 1
                # if started in starvation
                if stv2no[0] < no2stv[0]:
                    no2stv = np.append([0],no2stv)
                # adjust lengths
                newlen = min(stv2no.size, no2stv.size)
                stv2no = stv2no[:newlen]
                no2stv = no2stv[:newlen]
                # compute lengths
                return (stv2no - no2stv, np.mean(stv2no - no2stv))
            else:
                return (np.nan, np.nan)
        except:
                return (np.nan, np.nan)

    def compute_starvation_lengths(self):
        """Compute the lengths of starvation periods."""
        self["starvation_lengths"],self["avg_starvation_length"] =\
                self.compute_flag_lengths(reverse=False)

    def compute_growth_lengths(self):
        """Compute the lengths of growth periods."""
        self["growth_lengths"],self["avg_growth_length"] =\
                self.compute_flag_lengths(reverse=True)

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
        #ss = self["snapshot_generations" if self["auto"] else "snapshot_stages"]
        ss = self["snapshot_stages"]
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
        # True values
        mean_repr = np.zeros(self["prob_mean"]["surv"].shape)
        mean_repr[:,self["maturity"]:] = self["prob_mean"]["repr"]
        self["mean_repr"] = mean_repr
        # Junk values
        q = np.mean(self["junk_mean"]["repr"], 1)
        junk_repr = np.tile(q[:,np.newaxis], [1,self["max_ls"]])
        junk_repr[:,:self["maturity"]] = 0
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

    # OBSERVED SURVIVAL, REPRODUCTION, FITNESS

    def compute_kaplan_meier(self):
        """Compute Kaplan-Meier survival curves."""
        if self["age_dist_N"] == "all":
            agedist = self["age_distribution"]
            popn = self["population_size"]
            agehist = np.repeat(popn, agedist.shape[1]).reshape(agedist.shape)*agedist
            dividend = agehist[1:,1:]
            divisor = np.copy(agehist[:-1,:-1])
            # flag division by zero
            divisor[divisor==0] = np.nan
            res = dividend / divisor
            # substitute flags for ones
            res[np.isnan(res)] = 1
            self["kaplan-meier"] = np.concatenate((np.ones(1),np.mean(np.cumprod(res,1),0)))
        else:
            self.truncate_age_dist()
            m = self["age_distribution"].size/self["n_snapshots"]/self["max_ls"]
            ix = self["age_dist_stages"].flatten()
            popn = self["population_size"][ix].reshape(\
                    (self["n_snapshots"], m, 1))
            agehist = self["age_distribution"] * popn
            dividend = agehist[:,1:,1:]
            divisor = np.copy(agehist[:,:-1,:-1])
            # flag division by zero
            divisor[divisor==0] = np.nan
            res = dividend / divisor
            # substitute flags for ones
            res[np.isnan(res)] = 1
            self["kaplan-meier"] = np.concatenate((np.ones((res.shape[0],1)),\
                    np.mean(np.cumprod(res,2),1)), axis = 1)

    def compute_obs_repr(self):
        """Compute observed reproduction rate per age."""
        if self["age_dist_N"] == "all":
            self["observed_repr_rate"] = self["observed_repr_rate"].mean(0)
        else:
            self.truncate_age_dist()
            if self["age_dist_stages"].size > 0:
                self["observed_repr_rate"] = self["observed_repr_rate"].mean(1)

    def compute_obs_fitness(self):
        """Compute fitness using Kaplan-Meier survival and observed reproduction rates."""
        self["observed_fitness_term"] = self["kaplan-meier"]*self["observed_repr_rate"]
        if self["age_dist_N"] == "all":
            self["observed_fitness"] =np.sum(self["observed_fitness_term"])
        else:
            self["observed_fitness"] =np.sum(self["observed_fitness_term"],1)

    # MEAN AND VARIANCE IN BIT VALUE

    def compute_bits(self, pop):
        """Compute the mean and variance bit value at each position along the
        chromosome, sorted according to genmap_argsort."""
        """During finalisation, compute the distribution of 1s and 0s at each
        position on the chromosome (sorted by genome map), along with
        associated statistics."""
        l,m,b = self["max_ls"], self["maturity"], self["n_base"]
        # Reshape genomes to stack chromosomes
        # [snapshot, individual, bit]
        sc = pop.genomes.reshape(pop.N*2,pop.chr_len)
        # Compute order of bits in genome map
        order = np.ndarray.flatten(
            np.array([pop.genmap_argsort*b + c for c in xrange(b)]), order="F")
        # Average across individuals and sort [snapshot, bit]
        n1 = np.mean(sc, axis=0)[order]
        n1_var = np.var(sc, axis=0)[order]
        return n1, n1_var

    def compute_bits_snaps(self):
        """Wrapper of compute_bits for snapshot pops."""
        res = np.array([self.compute_bits(p) for p in self["snapshot_pops"]])
        self["n1"] = res[:,0]
        self["n1_var"] = res[:,1]

    def reorder_bits(self):
        """Reorder n1 record entry to original positions in genome."""
        if not "n1" in self.keys(): return
        # fix genmap for offset
        genmap = copy.deepcopy(self["genmap"])
        rofs = self["repr_offset"]
        nofs = self["neut_offset"]
        maxls = self["max_ls"]
        m = self["maturity"]
        ixr = np.logical_and(genmap>rofs, genmap<nofs)
        genmap[ixr] = genmap[ixr]-rofs-m+maxls
        ixn = genmap>=nofs
        genmap[ixn] = genmap[ixn]-nofs+2*maxls-m
        # reshape n1 so that it can be sorted
        nsnap = self["n_snapshots"]
        nb = self["n_base"]
        n1s = copy.deepcopy(self["n1"])
        n1s = n1s.reshape((nsnap,n1s.shape[1]/nb,nb))
        # sort n1
        n1s = n1s[:,genmap]
        # reshape back
        n1s = n1s.reshape((nsnap,n1s.shape[1]*nb))
        self["genmap_ix"] = genmap
        self["n1_reorder"] = n1s

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

    # SLIDING WINDOWS

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

    # TRUNCATION

    def truncate_age_dist_stages(self):
        # Truncate to taken snapshots
        self["age_dist_stages"] = self["age_dist_stages"][:self["n_snapshots"]]
        # If auto, make all sublists of same length
        if self["auto"]:
            # If empty, do nothing
            if not self["age_dist_stages"][0]:
                self["age_dist_stages"] = np.array(self["age_dist_stages"])
                return
            # If only one sublist, return numpy array
            if self["n_snapshots"]==1:
                self["age_dist_stages"] = np.array(self["age_dist_stages"])
                return
            # Find min length
            minl = len(self["age_dist_stages"][0])
            for sublist in self["age_dist_stages"][1:]:
                if sublist: minl = min(minl, len(sublist))
            # Find means
            means = [self["age_dist_stages"][0][0]]
            for sublist in self["age_dist_stages"][1:-1]:
                if sublist: means.append(np.mean(sublist).astype(int))
            last = False
            if self["age_dist_stages"][-1]:
                means.append(self["age_dist_stages"][-1][-1])
                last = True
            # Truncate
            self["age_dist_stages"] = make_windows(means, minl, last)
        else:
            self["age_dist_stages"] = np.array(self["age_dist_stages"])

    def truncate_age_dist(self, trunc_stages=True):
        """Truncate age distribution to nonzero entries."""
        if self["age_dist_truncated"]: return
        if self["age_dist_N"] == "all":
            self["age_dist_truncated"] = True
            return
        if trunc_stages: self.truncate_age_dist_stages()
        if self["age_dist_stages"].size > 0:
            for key in ["age_distribution","observed_repr_rate"]:
                ix = self["age_dist_stages"].flatten()
                self[key] = self[key][ix]
                # reshape to dim=(snapshot, stage, age)
                m = self[key].size / self["n_snapshots"] / self["max_ls"]
                self[key] = np.reshape(self[key],\
                    (self["n_snapshots"], m, self["max_ls"]))
        self["age_dist_truncated"] = True

    def truncate_per_stage_entries(self):
        """If autostage truncate per-stage entries to those where population size is
        recorded."""
        if self["auto"]:
            per_stage_entries = ["population_size",\
                                 "resources",\
                                 "starvation_flag",\
#                                 "bit_variance",\
                                 "age_distribution",\
                                 "observed_repr_rate",\
                                 "generation_dist",\
                                 "gentime_dist"]

            which = self["population_size"]>0
            for key in per_stage_entries:
                self[key] = self[key][which]

    # OVERALL

    def finalise(self, post_trunc=True):
        """Calculate additional stats from recorded data of a completed run."""
        # If dieoff, truncate data to last snapshot pop
        if self["dieoff"] or (self["auto"] and self["population_size"][-1]):
            pops = np.array(self["snapshot_pops"])
            pops = pops[np.nonzero(pops)]
            self["snapshot_pops"] = list(pops)
            self["n_snapshots"] = len(self["snapshot_pops"])
            if self["auto"]:
                self["snapshot_generations"] = self["snapshot_generations"][\
                    :self["n_snapshots"]]
            self["snapshot_stages"] = self["snapshot_stages"][:self["n_snapshots"]]
        # Truncate per-stage entries to recorded entries if autostage
        self.truncate_per_stage_entries()
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
        self.compute_starvation_lengths()
        self.compute_growth_lengths()
        self.compute_bits_snaps()
        self.compute_entropies()
        self.compute_windows()
        if post_trunc:
            self.compute_snapshot_age_dist_avrg()
            self.compute_kaplan_meier()
            self.compute_obs_repr()
            self.compute_obs_fitness()
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
