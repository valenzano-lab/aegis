# TODO investigate the calculation of fitness
# TODO plot fitness
# TODO improve visuals
# TODO universalize naming

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cPickle as pickle
import argparse, os

parser = argparse.ArgumentParser(description='Load record and plot\
        ')
parser.add_argument('dir', help="path to simulation directory")
parser.add_argument('-out', metavar="<str>", default=".", help="path to output directory")
parser.add_argument('-f', metavar="<str>", default="run_1_rec.txt",
        help="path to record file within dir (default: run_1_rec.txt)")

args = parser.parse_args()

os.chdir(args.dir)
recfile = open(args.f)
L = pickle.load(recfile) # dict
L["maxls"] = 71
L["n_stages"] = len(L["population_size"])
L["num_plots"] = len(L["snapshot_stages"])
L["var"] = True # variable resources
L["snapshot_stages"] = L["snapshot_stages"].astype(int) # convert to int so using as index is possible

outdir = "/home/asajina/local/genome-simulation/figures"

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
#    "fitness" : Average population fitness as predicted from genotypes
#    "entropy" : Shannon-Weaver entropy across entire population array
#    "junk_death" : Average death probability as predicted from neutral locus
#    "junk_repr"  : Average reproductive probability as predicted from neutral locus
#    "fitness" : Average population fitness as predicted from genotypes
#    "age_wise_fitness_contribution" : summands of fitness
#    "junk_age_wise_fitness_contribution" : summands of junk_fitness

######################
### PLOT FUNCTIONS ###
######################

# survival and standard deviation (2x1)
# colors represent values for snapshot stages with red being the most
# recent one, the blue line represents junk values for the most recent
# stage, vertical line is maturation age and the green line represents
# standard deviation
def survival(out=outdir):
    colormap = plt.cm.YlOrRd # select colormap
    colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["num_plots"])] # subset colormap
    m = L["maturity"]
    y_bound = ((1-L["d_range"])[0], (1-L["d_range"])[-1])
    x_bound = (0, L["maxls"])

    fig, ax = plt.subplots(2, sharex=True)

    # subplot 0
    for i in range(L["num_plots"]):
        ax[0].plot(1-L["death_mean"][i], color=colors[i]) # survival
    ax[0].plot((m,m),(y_bound[0],1), "k-") # maturity
    ax[0].plot(x_bound,(1-L["junk_death"][L["num_plots"]-1],1-L["junk_death"][L["num_plots"]-1]), "b-") # junk
    ax[0].set_ylim(y_bound)
    ax[0].set_xlim(x_bound)

    # subplot 1
    ax[1].plot(L["death_sd"][L["num_plots"]-1], color="green") # standard deviation
    ax[1].plot((m,m),(0,1), "k-") # maturity
    ax[1].set_ylim((0,max(L["death_sd"][L["num_plots"]-1])))
    # no need to set xlim because it's shared with subplot 0

    ax[0].set_ylabel("s", rotation="horizontal")
    ax[1].set_ylabel("s_sd", rotation="horizontal")
    ax[1].set_xlabel("age")
    fig.suptitle("Survival")
    plt.savefig(out+"/survival.png")
    plt.close()

# reproduction and standard deviation (2x1)
# colors represent values for snapshot stages with red being the most
# recent one, the blue line represents junk values for the most recent
# stage, vertical line is maturation age and the green line represents
# standard deviation
def reproduction(out=outdir):
    colormap = plt.cm.YlOrRd # select colormap
    colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["num_plots"])] # subset colormap
    m = L["maturity"]
    y_bound = ((L["r_range"])[0], (L["r_range"])[-1])
    x_bound = (L["maturity"], L["maxls"])

    fig, ax = plt.subplots(2, sharex=True)

    # subplot 0
    for i in range(L["num_plots"]):
        ax[0].plot(L["repr_mean"][i], color=colors[i]) # reproduction
    ax[0].plot(x_bound,(1-L["junk_repr"][L["num_plots"]-1],1-L["junk_repr"][L["num_plots"]-1]), "b-") # junk
    ax[0].set_ylim(y_bound)
    ax[0].set_xlim(x_bound)

    # subplot 1
    ax[1].plot(L["repr_sd"][L["num_plots"]-1], color="green") # standard deviation
    ax[1].set_ylim((0,max(L["repr_sd"][L["num_plots"]-1])))
    # no need to set xlim because it's shared with subplot 0

    ax[0].set_ylabel("r", rotation="horizontal")
    ax[1].set_ylabel("r_sd", rotation="horizontal")
    ax[1].set_xlabel("age")
    fig.suptitle("Reproduction")
    plt.savefig(out+"/reproduction.png")
    plt.close()

# population (blue) and resources (red)
# plot from stage s1 to stage s2, default: whole run
def pop_res(s1=0, s2=L["n_stages"], out=outdir):
    if L["var"]: l1,l2 = plt.plot(L["resources"][s1:s2+1],'r-',L["population_size"][s1:s2+1],'b-')
    else: l2,l1 = plt.plot(L["population_size"][s1:s2+1],'b-',L["resources"][s1:s2+1],'r-')

    plt.figure(1).legend((l1,l2),('resources','population'),'upper right',prop={'size':7})
    plt.title('Resources and population')
    plt.xlabel('stage')
    plt.ylabel('N',rotation='horizontal')
    plt.axis([s1,s2,0,max(max(L["resources"][s1:s2+1]),max(L["population_size"][s1:s2+1]))])
    plt.xticks(np.linspace(0,s2-s1,6),map(str,(np.linspace(s1,s2,6)).astype(int)))

    plt.savefig(out+'/pop_res.png')
    plt.close()

# age distribution
# colors represent values for snapshot stages with red being the most
# recent one
def age_distribution(out=outdir):
    colormap = plt.cm.YlOrRd # select colormap
    colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["num_plots"])] # subset colormap

    for i,j in zip(L["snapshot_stages"]-1,range(L["num_plots"])):
        plt.plot(L["age_distribution"][i], color=colors[j])

    plt.title('Age distribution')
    plt.xlabel('age')
    plt.ylabel('fraction',rotation='vertical')

    plt.savefig(out+'/age_distribution.png')
    plt.close()

# frequency of 1's
# (n1 is already sorted in age-ascending order, surv preceds repr)
# red lines mark maturation and where reproduction begins
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted with the recording standard deviation
def frequency(out=outdir, All=False):
    mv = (L["maturity"]*L["n_bases"], L["maturity"]*L["n_bases"])
    rv = (L["maxls"]*L["n_bases"], L["maxls"]*L["n_bases"])

    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["num_plots"]))

        for i,j,k in ix:
            ax[i,j].scatter(range(L["chr_len"]),L["n1"][k],s=5,c="k",marker=".")
            ax[i,j].plot(mv, (0,1), "r--")
            ax[i,j].plot(rv, (0,1), "r-")
            ax[i,j].xaxis.set_ticks([0,mv[0],rv[0]])
            ax[i,j].yaxis.set_ticks([0,1])
            ax[i,j].set_xticklabels([0,mv[0][0],rv[0][0]],fontsize=7)
            ax[i,j].set_yticklabels([0,1],fontsize=7)
            ax[i,j].set_ylim((0,1))
            ax[i,j].set_xlim((0,L["chr_len"]))
        fig.suptitle("Frequency of 1's")

    else:
        fig, ax = plt.subplots(2,sharex=True)

        # subplot 0
        ax[0].scatter(range(L["chr_len"]),L["n1"][-1],c="k")
        ax[0].plot(mv, (0,1), "r--")
        ax[0].plot(rv, (0,1), "r-")
        ax[0].xaxis.set_ticks([0,mv[0],rv[0]])
        ax[0].yaxis.set_ticks([0,1])
        ax[0].set_xlim((0,L["chr_len"]))
        ax[0].set_ylim((0,1))

        # subplot 1
        ax[1].plot(L["n1_std"][-1], "k-")
        ax[1].plot(mv, (0,1), "r--")
        ax[1].plot(rv, (0,1), "r-")
        ax[1].set_ylim((0,max(L["n1_std"][-1])))

        ax[0].set_ylabel("frequency", rotation="vertical")
        ax[1].set_ylabel("sd", rotation="vertical")
        ax[1].set_xlabel("position")
        fig.suptitle("Frequency of 1's")


    plt.savefig(out+"/frequency_1s.png")
    plt.close()

# frequency of 1's
# (age_wise_n1 is already sorted in age-ascending order, surv preceds repr)
# red lines mark maturation and where reproduction begins
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted with the recording standard deviation
def age_wise_frequency(out=outdir, All=False):
    mv = (L["maturity"], L["maturity"])
    rv = (L["maxls"], L["maxls"])

    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["num_plots"]))

        for i,j,k in ix:
            ax[i,j].scatter(range(L["chr_len"]/L["n_bases"]),L["age_wise_n1"][k],s=7,c="k",marker=".")
            ax[i,j].plot(mv, (0,1), "r--")
            ax[i,j].plot(rv, (0,1), "r-")
            ax[i,j].xaxis.set_ticks([0,mv[0],rv[0]])
            ax[i,j].yaxis.set_ticks([0,1])
            ax[i,j].set_xticklabels([0,mv[0][0],rv[0]],fontsize=7)
            ax[i,j].set_yticklabels([0,1],fontsize=7)
            ax[i,j].set_ylim((0,1))
            ax[i,j].set_xlim((0,L["chr_len"]/L["n_bases"]))
        fig.suptitle("Age-wise frequency of 1's")

    else:
        fig, ax = plt.subplots(2,sharex=True)

        # subplot 0
        ax[0].scatter(range(L["chr_len"]/L["n_bases"]),L["age_wise_n1"][-1],c="k")
        ax[0].plot(mv, (0,1), "r--")
        ax[0].plot(rv, (0,1), "r-")
        ax[0].xaxis.set_ticks([0,mv[0],rv[0]])
        ax[0].yaxis.set_ticks([0,1])
        ax[0].set_xlim((0,L["chr_len"]/L["n_bases"]))
        ax[0].set_ylim((0,1))

        # subplot 1
        ax[1].plot(L["age_wise_n1_std"][-1], "k-")
        ax[1].plot(mv, (0,1), "r--")
        ax[1].plot(rv, (0,1), "r-")
        ax[1].set_ylim((0,max(L["age_wise_n1_std"][-1])))

        ax[0].set_ylabel("frequency", rotation="vertical")
        ax[1].set_ylabel("sd", rotation="vertical")
        ax[1].set_xlabel("age")
        fig.suptitle("Age-wise frequency of 1's")

    plt.savefig(out+"/age_wise_frequency_1s.png")
    plt.close()

# density of genotypes (age unspecific)
# survival (green) and reproduction (red)
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted
def density(out=outdir, All=False):
    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["num_plots"]))
        y_bound = (0,np.around(max(np.max(L["density_surv"]),np.max(L["density_repr"])),2))

        for i,j,k in ix:
            ax[i,j].plot(L["density_surv"][k],"g-")
            ax[i,j].plot(L["density_repr"][k],"r-")
            ax[i,j].set_ylim(y_bound)
            ax[i,j].yaxis.set_ticks(np.linspace(y_bound[0],y_bound[1],5))
            #ax[i,j].set_xlim((0,L["chr_len"]/L["n_bases"]))
        fig.suptitle("Age-wise frequency of 1's")

    else:
        plt.plot(L["density_surv"][-1],"g-")
        plt.plot(L["density_repr"][-1],"r-")

        plt.ylabel("fraction", rotation="vertical")
        plt.xlabel("genotype")
        plt.title("Density")

    plt.savefig(out+"/density.png")
    plt.close()

# observed death rate (calculated from population_size*age_distribution)
# averaged over s1:s2, default: last 100 stages
def observed_death_rate(out=outdir, s1=L["n_stages"]-100, s2=L["n_stages"]):
    plt.plot(np.mean(L["actual_death_rate"][s1:s2],0))
    plt.ylabel(r"$\mu$", rotation="horizontal")
    plt.xlabel("age")
    plt.title("Observed death rate")

    plt.savefig(out+"/observed_death_rate.png")
    plt.close()

# entropy (shannon diversity index)
def shannon_diversity(out=outdir):
    plt.plot(L["entropy"], "o-")
    plt.ylabel("H", rotation="horizontal")
    plt.xlabel("stage")
    plt.title("Shannon diversity index")

    plt.savefig(out+"/shannon_diversity.png")
    plt.close()

# fitness (value per snapshot)
def fitness(out=outdir):
    fig,ax = plt.subplots()
    ax.plot(L["fitness"], "o-")
    ax.set_ylabel("F", rotation="horizontal")
    ax.set_xlabel("stage")
    ax.set_xticklabels(L["snapshot_stages"][::2])
    plt.title("Fitness")

    plt.savefig(out+"/fitness.png")
    plt.close()

# age wise fitness contribution
def age_wise_fitness_contribution(out=outdir, All=False):
    mv = (L["maturity"], L["maturity"])

    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["num_plots"]))

        for i,j,k in ix:
            ax[i,j].plot(L["age_wise_fitness_contribution"][k])
            ax[i,j].xaxis.set_ticks([0,mv[0],L["maxls"]-1])
            ax[i,j].set_xticklabels([0,mv[0][0],L["maxls"]-1],fontsize=7)
            #ax[i,j].set_ylim((0,np.max(L["age_wise_fitness_contribution"])))
            ax[i,j].set_xlim((L["maturity"],L["maxls"]-1))
        fig.suptitle("Age-wise fitness contribution")

    else:
        fig, ax = plt.subplots()

        l1 = ax.plot(L["age_wise_fitness_contribution"][-1])
        l2 = ax.plot(L["junk_age_wise_fitness_contribution"][-1], "g-")
        #ax.xaxis.set_ticks([0,mv[0],L["maxls"]-1])
        ax.set_xlim((L["maturity"],L["maxls"]-1))

        blue_proxy = mpatches.Patch(color="blue", label='$F_{age}$')
        green_proxy = mpatches.Patch(color="green", label='junk $F_{age}$')
        ax.legend(handles=[blue_proxy,green_proxy],loc="upper right",prop={'size':7})
        ax.set_ylabel("$F_{age}$", rotation="horizontal")
        ax.set_xlabel("age")
        plt.title("Age-wise fitness contribution")

    plt.savefig(out+"/age_wise_fitness_contribution.png")
    plt.close()


#survival()
#reproduction()
#pop_res()
#age_distribution()
#frequency()
#age_wise_frequency()
#density()
#observed_death_rate()
#shannon_diversity()
#print L["repr_mean"][-1]

#a = 1-L["junk_death"]
#print a.shape
#print a, "\n"
#b = np.tile(a.reshape(a.shape[0],1),2)
#print b.shape
#print b

#print L["fitness"].shape
#print L["age_wise_fitness_contribution"].shape
#print L["junk_age_wise_fitness_contribution"].shape

fitness()
age_wise_fitness_contribution(All=False)
