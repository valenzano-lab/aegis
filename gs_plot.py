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

L = get_record(args.file, args.r).record
O = get_outdir(args.o, args.file)
lns = L["n_snapshots"]

# plotting variables (not meant for UI)
tick_size = 7

### FULL LIST OF RECORD ITEMS ###
#    "n_bases" : number of bases making up one genetic unit
#    "maturity" : age at which sexual maturity is reached
#    "gen_map" : genome map for the run
#    "chr_len" : length of each chromosome in bits
#    "d_range" : range of possible death probabilities, from max to min
#    "r_range" : range of possible reproduction probabilities (min->max)
#    "snapshot_stages" : stages of run at which detailed info recorded
#    "population_size" : Value of N
#    "resources" : Resource level
#    "starvation_factor" : Value of x
#    "age_distribution" : Proportion of population at each age
#    "death_mean" : Mean genetic death probability at each age
#    "death_sd" : SD generic death probability at each age
#    "actual_death_rate" : per-age stage-to-stage fraction of survivors
#    "repr_mean" : Mean reproductive probability at each age
#    "repr_sd" : Mean reproductive probability at each age
#    "density_surv" : Distribution of number of 1's at survival loci
#    "density_repr" : Distribution of number of 1's at reproductive loci
#    "n1" : Average number of 1's at each position along the length of the chromosome
#    "n1_std" : n1 standard deviation
#    "age_wise_n1" : n1 averaged in intervals of n_bases
#    "age_wise_n1_std" : age_wise_n1 standard deviation
#    "s1" : Sliding-window SD of number of 1's along chromosome
#    "entropy" : Shannon-Weaver entropy across entire population array
#    "junk_death" : Average death probability as predicted from neutral locus
#    "junk_repr"  : Average reproductive probability as predicted from neutral locus
#    "fitness" : Average population fitness as defined in our article
#    "age_wise_fitness_product" : fitness calculated as s_i*r_i
#    "junk_age_wise_fitness_product" : fitness calculated as js_i*jr_i
#    "age_wise_fitness_contribution" : summands of fitness
#    "junk_age_wise_fitness_contribution" : summands of junk_fitness
#! Update this

######################
### PLOT FUNCTIONS ###
######################
# Get and subset colour map
colormap = plt.cm.YlOrRd
colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["n_snapshots"])]

# survival and standard deviation (2x1)
# colors represent values for snapshot stages with red being the most
# recent one, the blue line represents junk values for the most recent
# stage, vertical line is maturation age and the green line represents
# standard deviation

def save_close(name): 
    plt.savefig(os.path.join(O, name + ".png"))
    plt.close()

def set_axis_labels(ax, ylab, xlab, xticks):
    if not isinstance(ylab, list):
        ax.set_ylabel(ylab, rotation="vertical")
        ax.set_xlabel(xlab)
    else:
        for x in xrange(len(ylab)):
            if ylab[x] != "": 
                ax[x].set_ylabel(ylab[x], rotation="vertical")
            if xlab[x] != "":
                ax[x].set_xlabel(xlab[x])
    if xticks != "": ax.set_xticklabels(xticks)
    return ax

def survival():
    """Plot mean and SD of age-wise survival probability, superposing curves
    from different snapshot stages (red = most recent), as well as most-recent
    junk values (blue line) and maturity (black vertical line)."""
    m,x_bound = L["maturity"], (0, L["max_ls"])
    y_bound = ((1-L["d_range"])[0], (1-L["d_range"])[-1])
    fig, ax = plt.subplots(2, sharex=True)
    # Subplot 0 - mean survival
    for i in range(L["n_snapshots"]):
        ax[0].plot(1-L["death_mean"][i], color=colors[i])
    ax[0].plot((m,m),(y_bound[0],1), "k-") # maturity line
    ax[0].plot(x_bound,(1-L["junk_death"][L["n_snapshots"]-1],
        1-L["junk_death"][L["n_snapshots"]-1]), "b-") # junk
    ax[0].set_ylim(y_bound)
    ax[0].set_xlim(x_bound)
    # Subplot 1 - SD survival
    ax[1].plot(L["death_sd"][L["n_snapshots"]-1], color="green")
    #! Only most recent SD?
    ax[1].plot((m,m),(0,1), "k-") # maturity line
    ax[1].set_ylim((0,max(L["death_sd"][L["n_snapshots"]-1])))
    # no need to set xlim because it's shared with subplot 0
    ax = set_axis_labels(ax, ["s","s_sd"], ["","age"], "")
    fig.suptitle("Survival")
    save_close("survival")

def reproduction():
    """Plot mean and SD of age-wise reproduction probability, superposing 
    curves from different snapshot stages (red = most recent), as well as 
    most-recent junk values (blue line)."""
    m,x_bound = L["maturity"],(L["maturity"], L["max_ls"])
    y_bound = ((L["r_range"])[0], (L["r_range"])[-1])
    fig, ax = plt.subplots(2, sharex=True)
    # Subplot 0 - mean reproduction
    for i in range(L["n_snapshots"]):
        ax[0].plot(L["repr_mean"][i], color=colors[i])
    ax[0].plot(x_bound,(L["junk_repr"][L["n_snapshots"]-1],
        L["junk_repr"][L["n_snapshots"]-1]), "b-") # junk
    ax[0].set_ylim(y_bound)
    ax[0].set_xlim(x_bound)
    #ax[0].yaxis.set_ticks(np.linspace(L["r_range"][0],L["r_range"][-1],5))
    # Subplot 1 - SD reproduction
    ax[1].plot(L["repr_sd"][L["n_snapshots"]-1], color="green") # SD
    ax[1].set_ylim((0,max(L["repr_sd"][L["n_snapshots"]-1])))
    # no need to set xlim because it's shared with subplot 0
    ax = set_axis_labels(ax, ["r","r_sd"], ["", "age"], "")
    fig.suptitle("Reproduction")
    save_close("reproduction")

def pop_res(s1=0, s2=L["n_stages"]):
    """Plot population (blue) and resources (res) in specified stage range."""
    if L["res_var"]: # based on L["var"], plot either res or pop_size on top
        l1,l2 = plt.plot(L["resources"][s1:s2+1],"r-",
                L["population_size"][s1:s2+1],"b-")
    else: 
        l2,l1 = plt.plot(L["population_size"][s1:s2+1],"b-",
                L["resources"][s1:s2+1],"r-")
    plt.figure(1).legend((l1,l2),("resources","population"),
            "upper right",prop={"size":7})
    plt.title("Resources and population")
    plt.xlabel("stage")
    plt.ylabel("N",rotation="horizontal")
    plt.axis([s1,s2,0,max(max(L["resources"][s1:s2+1]),
        max(L["population_size"][s1:s2+1]))])
    plt.xticks(np.linspace(0,s2-s1,6),
            map(str,(np.linspace(s1,s2,6)).astype(int)))
    save_close("pop_res")

def age_distribution():
    """Plot age_distribution for each snapshot (red = most recent)."""
    for i,j in zip(L["snapshot_stages"]-1,range(L["n_snapshots"])):
        plt.plot(L["age_distribution"][i], color=colors[j])
    plt.title("Age distribution")
    plt.xlabel("age")
    plt.ylabel("fraction",rotation="vertical")
    save_close("age_distribution")

# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted with the recording standard deviation
def frequency(plot_all=False):
    """Plot the mean frequency of 1's at each bit position along the sorted
    genome map: juvenile survival, mature survival, mature reproduction. If
    plot_all, plot all snapshots on a grid; else plot the final snapshot
    along with the accompanying standard deviation."""
    mv = (L["maturity"]*L["n_bases"], 
            L["maturity"]*L["n_bases"]) # maturity vertical
    rv = (L["max_ls"]*L["n_bases"], 
            L["max_ls"]*L["n_bases"]) # reproduction vertical
    if plot_all:
        x = int(math.sqrt(lns))
        y = int(math.ceil(lns/float(x)))
        fig, ax = plt.subplots(x,y,sharex="col",sharey="row")
        ix = zip(np.sort(np.tile(range(x),x)), np.tile(range(y),y), range(lns))
        for i,j,k in ix:
            ax[i,j].scatter(range(L["chr_len"]),L["n1"][k],s=5,c="k",marker=".")
            ax[i,j].plot(mv, (0,1), "r--")
            ax[i,j].plot(rv, (0,1), "r-")
            ax[i,j].xaxis.set_ticks([0,mv[0],rv[0]])
            ax[i,j].set_xticklabels([0,mv[0][0],rv[0][0]],fontsize=tick_size)
            ax[i,j].set_yticklabels(np.linspace(0,1,6),fontsize=tick_size)
            ax[i,j].set_ylim((0,1))
            ax[i,j].set_xlim((0,L["chr_len"]))
        fig.text(0.03,0.55,"frequency",rotation="vertical",fontsize=12)
        fig.text(0.45,0.03,"position",rotation="horizontal",fontsize=12)
        fig.suptitle("Frequency of 1's (All Snapshots)")
        #! No SD when plotting all?
    else:
        fig, ax = plt.subplots(2,sharex=True)
        # Subplot 0
        ax[0].scatter(range(L["chr_len"]),L["n1"][-1],c="k")
        ax[0].plot(mv, (0,1), "r--")
        ax[0].plot(rv, (0,1), "r-")
        ax[0].xaxis.set_ticks([0,mv[0],rv[0]])
        ax[0].yaxis.set_ticks([0,1])
        ax[0].set_xlim((0,L["chr_len"]))
        ax[0].set_ylim((0,1))
        # Subplot 1
        ax[1].plot(L["n1_std"][-1], "k-")
        ax[1].plot(mv, (0,1), "r--")
        ax[1].plot(rv, (0,1), "r-")
        ax[1].set_ylim((0,max(L["n1_std"][-1])))
        ax = set_axis_labels(ax, ["frequency","sd"], ["", "position"], "")
        fig.suptitle("Frequency of 1's (Final Snapshot + SD)")
    filename = "frequency_1s_all" if plot_all else "frequency_1s_final"
    save_close(filename)

def age_wise_frequency(plot_all=False):
    """Plot the mean number of 1's in each locus along the sorted
    genome map: juvenile survival, mature survival, mature reproduction. If
    plot_all, plot all snapshots on a grid; else plot the final snapshot
    along with the accompanying standard deviation."""
    mv = (L["maturity"], L["maturity"]) # maturity vertical
    rv = (L["max_ls"], L["max_ls"]) # reproduction vertical
    if plot_all:
        x = int(math.sqrt(lns))
        y = int(math.ceil(lns/float(x)))
        fig, ax = plt.subplots(x,y,sharex="col",sharey="row")
        ix = zip(np.sort(np.tile(range(x),x)), np.tile(range(y),y), range(lns))
        for i,j,k in ix:
            ax[i,j].scatter(range(L["chr_len"]/L["n_bases"]),
                    L["age_wise_n1"][k],s=7,c="k",marker=".")
            ax[i,j].plot(mv, (0,1), "r--")
            ax[i,j].plot(rv, (0,1), "r-")
            ax[i,j].xaxis.set_ticks([0,mv[0],rv[0]])
            ax[i,j].set_xticklabels([0,mv[0][0],rv[0]],fontsize=tick_size)
            ax[i,j].set_yticklabels(np.linspace(0,1,6),fontsize=tick_size)
            ax[i,j].set_ylim((0,1))
            ax[i,j].set_xlim((0,L["chr_len"]/L["n_bases"]))
        fig.text(0.03,0.55,"frequency",rotation="vertical",fontsize=12)
        fig.text(0.45,0.03,"position",rotation="horizontal",fontsize=12)
        fig.suptitle("Age-wise frequency of 1's (all snapshots)")
        #! No SD when plotting all?
    else:
        fig, ax = plt.subplots(2,sharex=True)
        # Subplot 0
        ax[0].scatter(range(L["chr_len"]/L["n_bases"]),L["age_wise_n1"][-1],c="k")
        ax[0].plot(mv, (0,1), "r--")
        ax[0].plot(rv, (0,1), "r-")
        ax[0].xaxis.set_ticks([0,mv[0],rv[0]])
        ax[0].yaxis.set_ticks([0,1])
        ax[0].set_xlim((0,L["chr_len"]/L["n_bases"]))
        ax[0].set_ylim((0,1))
        # Subplot 1
        ax[1].plot(L["age_wise_n1_std"][-1], "k-")
        ax[1].plot(mv, (0,1), "r--")
        ax[1].plot(rv, (0,1), "r-")
        ax[1].set_ylim((0,max(L["age_wise_n1_std"][-1])))
        ax = set_axis_labels(ax, ["frequency","sd"], ["", "age"], "")
        fig.suptitle("Age-wise frequency of 1's (final snapshot + SD)") #! New name
    filename = "age_wise_frequency_1s_all" if plot_all \
            else "age_wise_frequency_1s_final"
    save_close(filename)

def density(plot_all=False):
    """Plot the distribution of genotypes (locus sums) across all survival
    (blue) and reproduction (red) loci in the genome. If plot_all, plot all 
    snapshots on a grid; else plot the final snapshot alone."""
    if plot_all:
        x = int(math.sqrt(lns))
        y = int(math.ceil(lns/float(x)))
        fig, ax = plt.subplots(x,y,sharex="col",sharey="row")
        ix = zip(np.sort(np.tile(range(x),x)), np.tile(range(y),y), range(lns))
        y_bound = (0,np.around(max(np.max(L["density_surv"]),
            np.max(L["density_repr"])),2))
        for i,j,k in ix:
            ax[i,j].plot(L["density_surv"][k],"b-")
            ax[i,j].plot(L["density_repr"][k],"r-")
            ax[i,j].set_ylim(y_bound)
            ax[i,j].yaxis.set_ticks(np.linspace(y_bound[0],y_bound[1],5))
            ax[i,j].tick_params(axis="both",labelsize=7)
        fig.text(0.03,0.55,"genotype",rotation="vertical",fontsize=12)
        fig.text(0.45,0.03,"density",rotation="horizontal",fontsize=12)
        fig.suptitle("Genotype density distribution (all snapshots)")
    else:
        plt.plot(L["density_surv"][-1],"b-")
        plt.plot(L["density_repr"][-1],"r-")
        plt.ylabel("fraction", rotation="vertical")
        plt.xlabel("genotype")
        plt.title("Genotype density distribution (final snapshot only)")
    filename = "density_all" if plot_all else "density_final"
    save_close(filename)

def observed_death_rate(s1=L["n_stages"]-100, s2=L["n_stages"]):
    """Plot the age-wise observed death rate for each stage within the
    specified limits."""
    #! Change default limits?
    plt.plot(np.mean(L["actual_death_rate"][s1:s2],0))
    plt.ylabel(r"$\mu$", rotation="horizontal")
    plt.xlabel("age")
    plt.title("Observed death rate")
    save_close("observed_death_rate")

def shannon_diversity():
    """Plot Shannon entropy of population gene pool per snapshot."""
    fig,ax = plt.subplots()
    ax.plot(L["entropy"], "o-")
    ax = set_axis_labels(ax, "H", "stage", L["snapshot_stages"][::2])
    plt.title("Shannon diversity index")
    save_close("shannon")

def fitness():
    """Plot population genomic fitness per snapshot."""
    fig,ax = plt.subplots()
    ax.plot(L["fitness"], "o-")
    ax = set_axis_labels(ax, "F", "stage", L["snapshot_stages"][::2])
    plt.title("Fitness")
    save_close("fitness")

def age_wise_fitness_product(plot_all=False):
    """Plot the mean survival probablity multiplied by the mean reproduction
    probability at each age. If plot_all, plot all snapshots on a grid; else
    plot the final snapshot and the corresponding neutral locus values"""
    if plot_all:
        x = int(math.sqrt(lns))
        y = int(math.ceil(lns/float(x)))
        fig, ax = plt.subplots(x,y,sharex="col",sharey="row")
        ix = zip(np.sort(np.tile(range(x),x)), np.tile(range(y),y), range(lns))
        for i,j,k in ix:
            #! Unify and functionalise grid-plotting code
            ax[i,j].plot(L["age_wise_fitness_product"][k])
            ax[i,j].set_xlim((L["maturity"],L["max_ls"]-1))
            ax[i,j].yaxis.set_major_locator(
                    ticker.MaxNLocator(5)) # set tick number to 5
            ax[i,j].tick_params(axis="both",labelsize=7)
        fig.text(0.01,0.52,"$prod \\ F_{age}$",rotation="horizontal",
                fontsize=12)
        fig.text(0.48,0.03,"age",rotation="horizontal",fontsize=12)
        fig.suptitle("Product fitness")
    else:
        fig, ax = plt.subplots()
        l1 = ax.plot(L["age_wise_fitness_product"][-1])
        l2 = ax.plot(L["junk_age_wise_fitness_product"][-1], "g-")
        ax.set_xlim((L["maturity"],L["max_ls"]-1))
        blue_proxy = mpatches.Patch(color="blue", label="product $F_{age})$")
        green_proxy = mpatches.Patch(color="green",
                label="junk product $F_{age}$")
        ax.legend(handles=[blue_proxy,green_proxy],loc="upper right",
                prop={"size":7})
        ax = set_axis_labels(ax, "product $F_{age}$", "age", "")
        plt.title("Product fitness")
    save_close("age_wise_fitness_product")

def age_wise_fitness_contribution(plot_all=False):
    """Plot the contribution to the total mean genomic fitness of the
    population by each age class in log space. If plot_all, plot all snapshots
    on a grid; else plot the final snapshot and the corresponding neutral locus
    values"""
    if plot_all:
        x = int(math.sqrt(lns))
        y = int(math.ceil(lns/float(x)))
        fig, ax = plt.subplots(x,y,sharex="col",sharey="row")
        ix = zip(np.sort(np.tile(range(x),x)), np.tile(range(y),y), range(lns))
        for i,j,k in ix:
            lawc = L["age_wise_fitness_contribution"][k]
            lawc,ages = lawc[lawc != 0], np.arange(L["max_ls"])[lawc != 0]
            ax[i,j].plot(ages,np.log10(lawc))
            ax[i,j].set_xlim((L["maturity"],L["max_ls"]-1))
            ax[i,j].yaxis.set_major_locator(
                    ticker.MaxNLocator(5)) # set tick number to 5
            ax[i,j].tick_params(axis="both",labelsize=7)
        fig.text(0.01,0.52,"$log(F_{age})$",rotation="horizontal",fontsize=12)
        fig.text(0.48,0.03,"age",rotation="horizontal",fontsize=12)
        fig.suptitle("Age-wise fitness contribution")
    else:
        lawc = L["age_wise_fitness_contribution"][-1]
        jlawc = L["junk_age_wise_fitness_contribution"][-1]
        ages, lawc = np.arange(L["max_ls"])[lawc != 0], lawc[lawc != 0],
        jages, jlawc = np.arange(L["max_ls"])[jlawc != 0], jlawc[jlawc != 0]
        fig, ax = plt.subplots()
        l1 = ax.plot(ages,np.log10(lawc))
        l2 = ax.plot(jages,np.log10(jlawc), "g-")
        ax.set_xlim((L["maturity"],L["max_ls"]-1))
        blue_proxy = mpatches.Patch(color="blue", label="$log(F_{age})$")
        green_proxy = mpatches.Patch(color="green",
                label="junk $log(F_{age})$")
        ax.legend(handles=[blue_proxy,green_proxy],
                loc="upper right",prop={"size":7})
        ax = set_axis_labels(ax, "log($F_{age}$)", "age", "")
        plt.title("Age-wise fitness contribution")
    save_close("age_wise_fitness_contribution")

def plot_all(pop_res_limits, odr_limits):
    """Generate all plots for the imported Record object."""
    print "Generating plots...",
    survival()
    reproduction()
    pop_res(pop_res_limits[0], pop_res_limits[1])
    age_distribution()
    observed_death_rate(odr_limits[0],odr_limits[1])
    shannon_diversity()
    fitness()
    for x in [True, False]:
        frequency(x)
        age_wise_frequency(x)
        density(x)
        age_wise_fitness_product(x)
        age_wise_fitness_contribution(x)
    print "done."

###############
### EXECUTE ###
###############

pop_res_limits = [0, L["n_stages"]] # Population/resources plot window
odr_limits = [L["n_stages"]-100, L["n_stages"]] # Observed death plot window
plot_all(pop_res_limits, odr_limits)
