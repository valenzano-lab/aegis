#!/usr/bin/env python
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
colormap = plt.cm.YlOrRd
colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["n_snapshots"])]

def save_close(name): 
    plt.tight_layout()
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
    if isinstance(values, list):
        for n in xrange(len(values)):
            if values[n] != "": getattr(ax[n], function)(values[n], **kwargs)
    elif values != "": getattr(ax, function)(values, **kwargs)

def axis_labels(ax, main, xlab, ylab):
    """Define the main title, x-axes and y-axes of one or more subplots in an 
    Axes object."""
    # Main title
    ax_iter(ax, "set_title", main, y=1.02)
    ax_iter(ax, "set_xlabel", xlab)
    ax_iter(ax, "set_ylabel", ylab, rotation="vertical")

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

def axis_legend(ax, cols, labels, loc="upper right", size=10):
    """Make a legend for an Axes object."""
    ax.legend(handles=make_handles(cols, labels), labels=labels,
            loc=loc, prop={"size":size})

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
    p,r,x = L["population_size"][s1:s2+1],L["resources"][s1:s2+1],L["res_var"]
    fig, ax = plt.subplots(1)
    if x: ax.plot(r,"r-",p,"b-")
    if not x: ax.plot(p,"b-",r,"r-")
    axis_labels(ax, "Resources and population", "Stage", "N")
    axis_ticks_limits(ax, np.linspace(0,s2-s1,6).astype(int), "", 
            (s1,s2), (0, max((max(r),max(p)))))
    make_legend(["blue", "red"], ["population", "resources"])
    save_close("1_pop_res")

# 2: STARVATION FACTORS
def starvation(limits=[0, L["n_stages"]]):
    """Plot starvation (green) and reproduction (magenta) starvation factors
    in specified stage range."""
    s1,s2 = limits
    r,s = L["repr_penf"][s1:s2+1],L["surv_penf"][s1:s2+1]
    fig, ax = plt.subplots(1)
    ax.plot(s,"g-",r,"m-")
    axis_labels(ax, "Starvation factors", "Stage", "Starvation factor")
    axis_ticks_limits(ax, np.linspace(0,s2-s1,6).astype(int), "", 
            (s1,s2), (0, max((max(r),max(s)))))
    make_legend(["green", "magenta"], ["survival","reproduction"])
    save_close("2_starvation")

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

# 4: GENOTYPE SUM WITH AGE
def genotype_sum():
    fig, ax = plt.subplots(1)
    for i in xrange(L["n_snapshots"]):
        gt = np.append(L["mean_gt"]["s"][i], L["mean_gt"]["r"][i])
        ax.plot(gt, color=colors[i])
    axis_labels(ax, "Mean genotype value with age", "Age",
            "Mean genotype(# of 1's in locus)")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("4_genotype_mean")
def genotype_var():
    fig, ax = plt.subplots(1)
    for i in xrange(L["n_snapshots"]):
        gt = np.append(L["var_gt"]["s"][i], L["var_gt"]["r"][i])
        ax.plot(gt, color=colors[i])
    axis_labels(ax, "Variance in genotype value with age", "Age",
            "Genotype variance")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("4a_genotype_var")

# 5: BIT VALUE WITH AGE
def bit_mean():
    fig, ax = plt.subplots(1)
    for i in xrange(L["n_snapshots"]):
        ax.plot(L["n1"][i], color=colors[i])
    axis_labels(ax, "Mean bit value with along sorted genome", "Bit position",
            "Average proportion of 1's")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("5_bit_mean")
def bit_var():
    fig, ax = plt.subplots(1)
    for i in xrange(L["n_snapshots"]):
        ax.plot(L["n1_var"][i], color=colors[i])
    axis_labels(ax, "Variance bit value with along sorted genome", "Bit position",
            "Variance in proportion of 1's")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("5a_bit_var")

# 6: GENOTYPE DENSITY
def density_overlay_surv():
    fig, ax = plt.subplots(1)
    d = L["density"]["s"].T
    for i in xrange(L["n_snapshots"]):
        ax.plot(d[i], color=colors[i])
    axis_labels(ax, "Distribution of survival genotypes at each snapshot",
            "Genotype sum", "Density")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("6a_density_surv")
def density_overlay_repr():
    fig, ax = plt.subplots(1)
    d = L["density"]["r"].T
    for i in xrange(L["n_snapshots"]):
        ax.plot(d[i], color=colors[i])
    axis_labels(ax, "Distribution of reproduction genotypes at each snapshot",
            "Genotype sum", "Density")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("6b_density_repr")

# 7: FITNESS
def fitness():
    fig, ax = plt.subplots(1)
    ax.plot(L["fitness"], "b-")
    axis_labels(ax, "Mean population fitness", "Snapshot", "Fitness")
    save_close("7_fitness")

# 8: FITNESS TERM
def fitness_term():
    fig, ax = plt.subplots(1)
    for i in xrange(L["n_snapshots"]):
        ax.plot(L["fitness_term"][i], color=colors[i])
    axis_labels(ax, "Age distribution in fitness contributions at each snapshot",
            "Age", "Fitness term")
    make_legend([colors[0], "white", colors[-1]], 
            ["Snapshot 1", "...", "Snapshot {}".format(lns)])
    save_close("8_fitness_term")

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
    save_close("9_observed_death")

def observed_death_old(s1=L["n_stages"]-100, s2=L["n_stages"]):
    """Plot the age-wise observed death rate for each stage within the
    specified limits."""
    #! Change default limits?
    plt.plot(np.mean(L["actual_death_rate"][s1:s2,:-1],0))
    plt.ylabel(r"$\mu$", rotation="horizontal")
    plt.xlabel("Age")
    plt.title("Observed death rate")
    save_close("9a_observed_death_old")

# 10: ENTROPY
def entropy():
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(L["entropy_bits"], "b-")
    ax[1].plot(L["entropy_gt"]["a"], "r-")
    axis_labels(ax[0], "Shannon entropy of population genomes", "", " ")
    axis_labels(ax[1], "", "Snapshot", " ")
    #axis_labels(ax, "Shannon entropy of population genomes at each snapshot", 
    #        "Snapshot", "")
    def make_legend(cols, labels, loc="upper right", size=10):
        ax[0].legend(handles=make_handles(cols, labels), labels=labels, loc=loc, 
                prop={"size":size})
    make_legend(["blue","red"], 
            ["Entropy in bit values", "Entropy in genotype sums"])
    fig.text(0, 0.5, 'H', va='center', rotation='vertical')
    save_close("10_entropy")

def plot_all(pop_res_limits, odr_limits):
    """Generate all plots for the imported Record object."""
    print "Generating plots...",
    pop_res(pop_res_limits)
    starvation(pop_res_limits)
    age_distribution()
    genotype_sum()
    genotype_var()
    bit_mean()
    bit_var()
    density_overlay_surv()
    density_overlay_repr()
    fitness()
    fitness_term()
    observed_death()
    observed_death_old()
    entropy()
    #observed_death_rate(odr_limits[0],odr_limits[1])
    #shannon_diversity()
    #fitness()
    #for x in [True, False]:
    #    frequency(x)
    #    age_wise_frequency(x)
    #    density(x)
    #    age_wise_fitness_product(x)
    #    age_wise_fitness_contribution(x)
    print "done."

###############
### EXECUTE ###
###############

pop_res_limits = [0, L["n_stages"]] # Population/resources plot window
odr_limits = [L["n_stages"]-100, L["n_stages"]] # Observed death plot window
plot_all(pop_res_limits, odr_limits)
