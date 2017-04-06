#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO universalize naming across files
# TODO maybe interpolate
# TODO update tests for fitness

# Libraries and arguments

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
import argparse,os,math,pyximport; pyximport.install()
from gs_core import Run, Simulation, Record

try:
       import cPickle as pickle
except:
       import pickle

parser = argparse.ArgumentParser(description="Load record and plot\
        the simulation data.")
parser.add_argument("file", help="path to file containing record data")
parser.add_argument("-o", metavar="<str>", default="", help="path to directory\
        in which to deposit figures (default: directory of target file)")
parser.add_argument("-r", metavar="<int>", default=-1, help="if target file \
        contains a simulation object with multiple runs, specify from which \
        run to take record data (default: average of all runs)")
args = parser.parse_args()

# Process I/O
def get_record(target, run=-1):
    """Retrieve a Record object from a pickled REC or SIM file."""
    if not os.path.isfile(target):
        print "No such file: " + target
        q = raw_input("Enter correct file path, or skip to abort: ")
        if q == "": exit("Aborting.")
        return self.get_startpop(q, run)
    print "Importing Record object from {}...".format(target),
    f = open(target, "rb")
    obj = pickle.load(f)
    print "done."
    if isinstance(obj, Record): return obj
    elif isinstance(obj, Run): return obj.record
    elif isinstance(obj, Simulation): 
        if run == -1: return obj.avg_record # Summary record
        else: return obj.runs[run].record # Record of specific run
    else: 
        x = "Inappropriate object type; cannot extract record data."
        raise TypeError(x)

def get_outdir(outpath, target):
    """Tests validity of output directory path and creates figure directory
    if necessary."""
    if outpath=="": outpath = os.path.dirname(target)
    if not os.path.isdir(outpath):
        print "No such directory: " + outpath
        q = raw_input("Enter path to existing directory, or skip to abort: ")
        if q == "": exit("Aborting.")
        return self.get_outdir(q, target)
    "Valid output directory: {}".format(outpath)
    figdir = os.path.join(outpath, "figures")
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    return figdir

L = get_record(args.file, int(args.r)).record
O = get_outdir(args.o, args.file)
lns = L["n_snapshots"]

# plotting variables (not meant for UI)
tick_size = 7

######################
### PLOT FUNCTIONS ###
######################

# Get and subset colour map
def get_colours(colourmap, rev=False, start=0.1, stop=0.9, n=L["n_snapshots"]):
    """Return evenly spaced colours according to a specified colourmap."""
    refs = np.linspace(start, stop, n)
    if rev: refs = refs[::-1]
    return [colourmap(i) for i in refs]
colors = get_colours(plt.cm.autumn, True)
colors2 = get_colours(plt.cm.summer, True)

def save_close(name): 
    plt.tight_layout()
    dirname = os.path.dirname(name)
    if dirname:
        dirpath = os.path.join(O,dirname)
        if not os.path.isdir(dirpath): os.mkdir(dirpath)
    plt.savefig(os.path.join(O, name + ".png"))
    plt.close()
def simple_plot_label(main, xlab, ylab):
    plt.title(main, y=1.02)
    plt.xlabel(xlab)
    plt.ylabel(ylab, rotation="vertical")

def ax_iter(ax, function, values, **kwargs):
    """Given an Axes object, a function name and one or more values, 
    apply that function to the Axes object with that value (if a single value),
    or apply it to each subplot in the object with the corresponding value
    (if a list of values)."""
    if isinstance(values, list) and isinstance(ax, np.ndarray):
        for n in xrange(len(values)):
            if values[n] != "": getattr(ax[n], function)(values[n], **kwargs)
    elif values != "": getattr(ax, function)(values, **kwargs)

def axis_labels(ax, main, xlab, ylab, xpad=None, ypad=None):
    """Define the main title, x-axes and y-axes of one or more subplots in an 
    Axes object."""
    # Main title
    ax_iter(ax, "set_title", main, y=1.02)
    ax_iter(ax, "set_xlabel", xlab, labelpad=xpad)
    ax_iter(ax, "set_ylabel", ylab, rotation="vertical", labelpad=ypad)

def axis_ticks_limits(ax, xticks, yticks, xlim, ylim):
    """Set the tick positions and axis limits for one or more subplots in an
    Axes object."""
    ax_iter(ax, "set_xticklabels", xticks)
    ax_iter(ax, "set_yticklabels", yticks)
    ax_iter(ax, "set_xlim", xlim)
    ax_iter(ax, "set_ylim", ylim)

def label_axes(ax, main, xlab, ylab):
    """Set the main, x-axis and y-axis labels for an Axes object."""
    ax.set_title(main, y=1.02)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab, rotation="vertical")

def axis_legend(ax, cols, labels, loc="upper right", size=10, **kwargs):
    """Make a legend for an Axes object."""
    ax.legend(handles=make_handles(cols, labels), labels=labels,
            loc=loc, prop={"size":size}, **kwargs)

def simple_plot(plot, legend_labels, main, xlab, ylab, axes, ticks, path):
    l1,l2 = plot
    plt.figure(1).legend((l1,l2), legend_labels, "upper right", prop={"size":7})
    simple_plot_label(main, xlab, ylab)
    plt.axis(axes)
    plt.xticks(ticks, map(str, ticks.astype(int)))
    save_close(path)

def finalise_plot(title, xlabel, ylabel, filename, handles=""):
    """Set axis and overall titles of a single plot, then save."""
    simple_plot_label(title, xlabel, ylabel)
    if handles=="":
        plt.legend(loc="upper right", prop={"size":10})
    else:
        plt.legend(handles=handles, loc="upper right", prop={"size":10})
    save_close(filename)

def make_handles(cols, labels):
    """Generate a list of handles for a figure legend."""
    if len(cols) != len(labels): 
        raise ValueError("Inconsistent colour and label inputs.""")
    handles = [0] * len(cols)
    for n in xrange(len(cols)):
        handles[n] = mpatches.Patch(color=cols[n], label=labels[n])
    return handles

def make_legend(cols, labels, loc="upper right", size=10):
    plt.legend(handles=make_handles(cols, labels), loc=loc, prop={"size":size})

def grid_plot(plot_func, title, xtitle, ytitle, filename):
    """Reproduce a plot once for each snapshot and arrange on a grid."""
    x = int(math.sqrt(lns))
    y = int(math.ceil(lns/float(x)))
    fig, ax = plt.subplots(x,y,sharex="col",sharey="row")
    ix = zip(np.sort(np.tile(range(x),x)), np.tile(range(y),y), range(lns))
    for i,j,k in ix:
        plot_func(ax[i,j], k)
    fig.text(0.45,0.03,xtitle,rotation="horizontal",fontsize=12)
    fig.text(0.03,0.55,ytitle,rotation="vertical",fontsize=12)
    fig.suptitle(title + " (all snapshots)")
    save_close(filename+"_all")

# 1: POPULATION & RESOURCES
def pop_res(limits=[0, L["n_stages"]]):
    """Plot population (blue) and resources (res) in specified stage range."""
    s1,s2 = limits
    print limits, L["population_size"].shape, L["resources"].shape
    print L["n_runs"]
    pop,res,nr = L["population_size"], L["resources"], L["n_runs"]
    pop = pop[:,s1:s2+1] if nr > 1 else pop[s1:s2+1]
    res = res[:,s1:s2+1] if nr > 1 else res[s1:s2+1]
    # TODO: Fix for single run
    cols, labels = ["blue", "red"], ["population", "resources"]
    main = "Population and resources"
    def make_pr_plot(nrun, cols, labels):
        p,r = pop[nrun], res[nrun]
        fig,ax = plt.subplots(1)
        if L["res_var"]:
            ax.plot(r,"r-",p,"b-")
            if hasattr(L, "res_regen_constant"): 
                ax.plot(L["res_regen_constant"], "m-")
                cols += ["magenta"]
                labels += ["R"]
        else: ax.plot(p,"b-",r,"r-")
        axis_labels(ax, main + " (Run {})".format(nrun+1), "Stage", "N")
        axis_ticks_limits(ax, np.linspace(0,s2-s1,6).astype(int), "", 
                (s1,s2), (0, max((max(r),max(p)))))
        make_legend(cols, labels)
        nzeros = len(str(L["n_runs"]+1))-len(str(nrun+1))
        save_close("1_pop_res/run"+"0"*nzeros+str(nrun+1))
    for n in xrange(L["n_runs"]): make_pr_plot(n, cols, labels)

# 2: STARVATION FACTORS
def starvation(limits=[0, L["n_stages"]]):
    """Plot starvation (green) and reproduction (magenta) starvation factors
    in specified stage range."""
    is_r, is_s = L["repr_pen"], L["surv_pen"] # Check if starvation occurs
    if not (is_r or is_s): return
    s1,s2 = limits
    rep,sur,nr = L["repr_penf"], L["surv_penf"], L["n_runs"]
    rep = rep[:,s1:s2+1] if nr > 1 else rep[s1:s2+1]
    sur = sur[:,s1:s2+1] if nr > 1 else sur[s1:s2+1]
    # TODO: Fix for single run
    is_rs = (is_r and is_s)
    suffix = "" if is_rs else " (Survival)" if is_s else " (Reproduction)"
    main,xlab,ylab="Starvation factors" + suffix, "Stage", "Starvation factor"
    def make_st_plot(nrun): # TODO: Can probably make this shorter
        r,s = rep[nrun], sur[nrun]
        fig,ax = plt.subplots(is_r + is_s)
        if is_rs:
            ax[0].plot(s,"g-")
            ax[1].plot(r,"m-")
            axis_labels(ax[0], main, "", "")
            axis_labels(ax[1], "", xlab, "")
            cols, labels = ["green", "magenta"],["survival","reproduction"]
            logbase_s = L["death_inc"] if hasattr(L, "death_inc") else 3
            logbase_r = L["repr_dec"] if hasattr(L, "repr_dec") else 3
            ax[0].set_yscale("log", basey=logbase_s)
            ax[1].set_yscale("log", basey=logbase_r)
            for n in xrange(len(ax)):
                axis_ticks_limits(ax[n], "", "", (s1,s2), "")
            for n in np.unique(L["surv_penf"]): # Check actual values of s-factor
                ax[0].axhline(n, color="k", linestyle="dashed")
            for n in np.unique(L["repr_penf"]):
                ax[1].axhline(n, color="k", linestyle="dashed")
            axis_legend(ax[0], cols, labels)
            fig.text(0.05,0.5,ylab,va="center",rotation="vertical")
        else:
            ax.plot(s if is_s else r, "g-" if is_s else "m-")
            axis_labels(ax, main, xlab, ylab)
            axis_ticks_limits(ax, "", "", (s1,s2), "")
            if is_s:
                logbase = L["death_inc"] if hasattr(L, "death_inc") else 3
            else:
                logbase = L["repr_dec"] if hasattr(L, "repr_dec") else 3
            ax.set_yscale("log", basey=logbase)
            for n in np.unique(L["surv_penf"] if is_s else L["repr_penf"]): 
                ax.axhline(n, color="k", linestyle="dashed")
        nzeros = len(str(L["n_runs"]+1))-len(str(nrun+1))
        save_close("2_starvation/run"+"0"*nzeros+str(nrun+1))
    for n in xrange(L["n_runs"]): make_st_plot(n)

# 3: AGE DISTRIBUTIONS
def age_distribution():
    """Plot age_distribution for each snapshot after the first."""
    fig, ax = plt.subplots(1)
    for i,j in zip(L["snapshot_stages"][1:],range(L["n_snapshots"])[1:]):
        ax.plot(L["age_distribution"][i]*100, color=colors[j])
    axis_labels(ax, "Age distribution","Age", "% of individuals")
    make_legend([colors[1], "white", colors[-1]], 
            ["Snapshot 2", "...", "Snapshot {}".format(lns)])
    save_close("3_age_distribution")

# 4: GENOTYPES WITH AGE
def genotypes_with_age():
    # Define values
    mean, var, maxstate = L["mean_gt"], L["var_gt"], L["n_states"]-1
    vals_mean = np.hstack((mean["s"],mean["r"]))
    vals_var = np.hstack((var["s"],var["r"]))
    nt_mean, nt_var = np.mean(mean["n"][-1]), np.mean(var["n"][-1])
    ls, mt, nloc = L["max_ls"]-1, L["maturity"], len(L["genmap"])-L["n_neutral"]
    # Make plots
    fig, ax = plt.subplots(2)
    for i in xrange(L["n_snapshots"]):
        ax[0].plot(vals_mean[i], color=colors[i])
        ax[1].plot(vals_var[i], color=colors2[i])
    for a in ax:
        a.axvline(mt, color="k") # Maturity line
        a.axvline(ls, color="k", linestyle="dashed") # Lifespan line
        a.xaxis.set_ticks([0, mt, ls, nloc-1])
        axis_ticks_limits(a, [0,mt,ls,ls], "", (0, nloc-1), "")
    ax[0].axhline(nt_mean, color="m", linestyle="dashed")
    ax[1].axhline(nt_var, color="c", linestyle="dashed")
    axis_labels(ax[0], "Mean and variance in genotype sum with age", "",
            "Mean genotype sum")
    axis_labels(ax[1], "", "Age", "Variance in genotype sum")
    axis_legend(ax[0], [colors[0], "white", colors[-1], "magenta"],
            ["Snapshot 1", "...", "Snapshot {}".format(lns),
                "S{} (Neutral)".format(lns)])
    axis_legend(ax[1], [colors2[0], "white", colors2[-1], "cyan"],
            ["Snapshot 1", "...", "Snapshot {}".format(lns),
                "S{} (Neutral)".format(lns)])
    save_close("4_genotypes_with_age")

# 4: BIT VALUES  WITH AGE
def bits_with_age():
    # Define values
    vals_mean, vals_var, nb = L["n1"], L["n1_var"], L["n_base"].astype(int)
    ls, mt, nn = nb*(L["max_ls"])-1, nb*L["maturity"], nb*L["n_neutral"].astype(int)
    nloc = L["chr_len"] - nn
    nt_mean, nt_var = np.mean(vals_mean[-1][-nn:]), np.mean(vals_var[-1][-nn:])
    # Make plots
    fig, ax = plt.subplots(2)
    for i in xrange(L["n_snapshots"]):
        ax[0].plot(vals_mean[i], color=colors[i])
        ax[1].plot(vals_var[i], color=colors2[i])
    for a in ax:
        a.axvline(mt, color="k") # Maturity line
        a.axvline(ls, color="k", linestyle="dashed") # Lifespan line
        a.xaxis.set_ticks([0, mt, ls, nloc-1])
        axis_ticks_limits(a, [0 , mt/nb, ls/nb, ls/nb], "", (0, nloc-1), "")
    ax[0].axhline(nt_mean, color="m", linestyle="dashed")
    ax[1].axhline(nt_var, color="c", linestyle="dashed")
    axis_labels(ax[0], "Mean and variance in bit value with age", "",
            "Mean bit value")
    axis_labels(ax[1], "", "Age", "Variance in bit value")
    axis_legend(ax[0], [colors[0], "white", colors[-1], "magenta"],
            ["Snapshot 1", "...", "Snapshot {}".format(lns),
                "S{} (Neutral)".format(lns)])
    axis_legend(ax[1], [colors2[0], "white", colors2[-1], "cyan"],
            ["Snapshot 1", "...", "Snapshot {}".format(lns),
                "S{} (Neutral)".format(lns)])
    save_close("5_bits_with_age")

# 6: GENOTYPE DENSITY
def density():
    # Define values
    s, r, n = L["density"]["s"].T, L["density"]["r"].T, L["density"]["n"].T
    n_colors = get_colours(plt.cm.Purples)
    y_max = np.max(np.vstack((s,r,n)))
    # Make plots
    fig, ax = plt.subplots(3)
    for i in xrange(L["n_snapshots"]):
        ax[0].plot(s[i], color=colors[i])
        ax[1].plot(r[i], color=colors2[i])
        ax[2].plot(n[i], color=n_colors[i])
    axis_labels(ax[0], "Genotype distributions at each snapshot", "",
            "Density (surv)")
    axis_labels(ax[1], "", "", "Density (repr)")
    axis_labels(ax[2], "", "Genotype sum", "Density (neut)")
    axis_legend(ax[0], [colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)],
            ncol=3)
    axis_legend(ax[1], [colors2[0], "white", colors2[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)],
            ncol=3)
    axis_legend(ax[2], [n_colors[0], "white", n_colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)],
            ncol=3)
    # TODO: Fix middle part of legend
    # Equalise y-axes
    for a in ax: a.set_ylim((0,y_max))
    save_close("6_density")

def density_difference():
    # Define values
    s, r, n = L["density"]["s"].T, L["density"]["r"].T, L["density"]["n"].T
    s, r = s-n, r-n
    #s, r = (s.T - n).T, (r.T - n).T
    # Make plots
    fig, ax = plt.subplots(2)
    for i in xrange(L["n_snapshots"]):
        ax[0].plot(s[i], color=colors[i])
        ax[1].plot(r[i], color=colors2[i])
    ax[0].axhline(0, color="k", linestyle="dashed")
    ax[1].axhline(0, color="k", linestyle="dashed")
    axis_labels(ax[0], "Difference in genotype distribution from neutral loci",
            "", "$\Delta$ Density (surv)")
    axis_labels(ax[1], "", "", "$\Delta$ Density (repr)")
    axis_legend(ax[0], [colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)],
            ncol=3)
    axis_legend(ax[1], [colors2[0], "white", colors2[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)],
            ncol=3)
    save_close("7_density_difference")

# 7: FITNESS
def fitness():
    fig, ax = plt.subplots(1)
    ax.plot(L["fitness"], "b-")
    axis_labels(ax, "Mean population fitness", "Snapshot", "Fitness")
    ax.xaxis.set_ticks(range(L["n_snapshots"]))
    axis_ticks_limits(ax,range(1,L["n_snapshots"]+1),"",(0,L["n_snapshots"]-1),"")
    save_close("8_fitness")

# 8: FITNESS TERM
def per_age_fitness():
    """Plot the per-age contribution to average genotypic fitness, equal to the
    expected number of offspring at that age."""
    fig, ax = plt.subplots(1)
    for i in xrange(L["n_snapshots"]):
        ax.plot(L["fitness_term"][i], color=colors[i])
    axis_labels(ax, "Expected number of offspring at each age and snapshot",
            "Age", "E(# of offspring)")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    axis_ticks_limits(ax, "","",(L["maturity"],L["max_ls"]-1),"") # Remove immature
    save_close("9_per_age_fitness")

def per_age_fitness_diff():
    """Plot the log-ratio of observed per-age contribution to fitness against
    that expected in the absence of ageing."""
    fig, ax = plt.subplots(1)
    mt,ls = L["maturity"].astype(int), L["max_ls"].astype(int)
    r,s = L["prob_mean"]["repr"][:,0][:,np.newaxis], L["prob_mean"]["surv"][:,mt]
    f_obs,x = L["fitness_term"][:,mt:], np.arange(mt,ls).astype(float)
    f_exp = (r * s **(x+1)) / (2 if L["sexual"] else 1)
    g = np.log10(f_obs/f_exp)
    for i in xrange(L["n_snapshots"]):
        ax.plot(g[i], color=colors[i])
    title = "Observed vs expected per-age fitness in the absence of ageing"
    axis_labels(ax, title, "Age", "$\log_{10}$ $(f_x^{obs}/f_x^{exp})$")
    ax.xaxis.set_ticks(range(ls-mt-1,0,-10)[::-1])
    axis_ticks_limits(ax, range(ls-1,mt,-10)[::-1],"",(0,ls-mt-1),"")
    axis_legend(ax, [colors[0], "white", colors[-1]],
            ["Snapshot 1", "...", "Snapshot {}".format(lns)], ncol=3)
    save_close("10_per_age_fitness_diff")

def ageing_index():
    """Plot the mean inter-age difference in observed vs expected per-age
    fitness as an overall measure of population ageing at each snapshot."""
    fig, ax = plt.subplots(1)
    mt,ls = L["maturity"].astype(int), L["max_ls"].astype(int)
    r,s = L["prob_mean"]["repr"][:,0][:,np.newaxis], L["prob_mean"]["surv"][:,mt]
    f_obs,x = L["fitness_term"][:,mt:], np.arange(mt,ls).astype(float)
    f_exp = (r * s **(x+1)) / (2 if L["sexual"] else 1)
    g = np.log10(f_obs/f_exp)
    h = - np.mean(np.diff(g, axis=1), axis=1)
    ax.plot(h, "b-")
    axis_labels(ax, "Measure of degree of ageing shown at each snapshot",
            "Snapshot", "Ageing index")
    ax.xaxis.set_ticks(range(L["n_snapshots"]))
    axis_ticks_limits(ax,range(1,L["n_snapshots"]+1),"",(0,L["n_snapshots"]-1),"")
    save_close("11_ageing_index")


# 9: OBSERVED DEATH RATE
def observed_death(width=100):
    """Plot average observed death rate around each snapshot."""
    # Derive plotting values
    def mean_od(s0,s1): 
        return np.mean(L["actual_death_rate"][s0:s1,:-1],0)
    S0 = L["snapshot_stages"] - int(width/2)
    S1 = L["snapshot_stages"] + int(width/2)
    S0[0],S1[0] = 0, width
    S0[-1],S1[-1] = L["n_stages"] - width, L["n_stages"]
    vals = np.array([mean_od(S0[n],S1[n]) for n in xrange(L["n_snapshots"])])
    # Plot values
    fig, ax = plt.subplots(1)
    for i in xrange(L["n_snapshots"]):
        ax.plot(vals[i], color=colors[i])
    axis_labels(ax, "Per-age observed death rate at each snapshot",
            "Age", "Death rate")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("12_observed_death")

# 10: ENTROPY
# TODO: Pad y axis
def entropy():
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(L["entropy_bits"], "b-")
    ax[1].plot(L["entropy_gt"]["a"], "r-")
    axis_labels(ax[0], "Shannon entropy of population genomes", "", " ", ypad=100)
    axis_labels(ax[1], "", "Snapshot", " ")
    axis_legend(ax[0], ["blue","red"], 
            ["Entropy in bit values", "Entropy in genotype sums"])
    fig.text(0, 0.5, 'H', va='center', rotation='vertical')
    ax[0].xaxis.set_ticks(range(L["n_snapshots"]))
    ax[1].xaxis.set_ticks(range(L["n_snapshots"]))
    axis_ticks_limits(ax[0],
            range(1,L["n_snapshots"]+1),"",(0,L["n_snapshots"]-1),"")
    axis_ticks_limits(ax[1],
            range(1,L["n_snapshots"]+1),"",(0,L["n_snapshots"]-1),"")
    save_close("13_entropy")

def plot_all(pop_res_limits, odr_limits):
    """Generate all plots for the imported Record object."""
    print "Generating plots...",
    pop_res(pop_res_limits)
    starvation(pop_res_limits)
    age_distribution()
    genotypes_with_age()
    bits_with_age()
    density()
    density_difference()
    fitness()
    per_age_fitness()
    per_age_fitness_diff()
    ageing_index()
    observed_death()
    entropy()
    print "done."

###############
### EXECUTE ###
###############

pop_res_limits = [0, L["n_stages"]] # Population/resources plot window
odr_limits = [L["n_stages"]-100, L["n_stages"]] # Observed death plot window
plot_all(pop_res_limits, odr_limits)
