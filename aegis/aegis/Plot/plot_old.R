#############
### USAGE ###
# Create a 'figures' directory in the simulation directory.
# In the EXECUTE section assign the simulation dir path to variable 'path' and
# call functions depending on what you want to plot.
# Be sure to first execute 'reshape data frames' and 'plot values' blocks.
##############

##############
### IMPORT ###
##############

library(rPython)
library(rjson)

get_record = function(path, run=1){
  # Imports record object from named directory and run as a Python dictionary object.
  # path = path to simulation directory from working directory
  # run = run number from simulation
  l0 <- "import cPickle,json"
  l1 <- paste("recfile = open('", path, "/run_", run, "_rec.txt', 'rb')", sep="")
  l2 <- "record = cPickle.load(recfile)"
  python.exec(c(l0, l1, l2))
}

get_item = function(item_name){
  # Returns named item from Python record object as a vector or matrix.
  str <- paste("json.dumps(record['",item_name,"'].tolist())", sep = "")
  json = python.get(str)
  item = fromJSON(json)
  if(is.list(item)) return(do.call(rbind, item))
  return(item)
}

get_item_list = function(){
  L = list(
    b = get_item("n_bases"),
    m = get_item("maturity"),
    maxls = get_item("max_ls"),
    gen_map = get_item("gen_map"),
    chr_len = get_item("chr_len"),
    d_range = get_item("d_range"),
    r_range = get_item("r_range"),
    snapshot_stages = get_item("snapshot_stages"),
    population_size = get_item("population_size"),
    resources = get_item("resources"),
    #starvation_factor = get_item("starvation_factor"),
    age_distribution = get_item("age_distribution"),
    death_mean = get_item("death_mean"),
    death_sd = get_item("death_sd"),
    actual_death_rate = get_item("actual_death_rate"),
    repr_mean = get_item("repr_mean"),
    repr_sd = get_item("repr_sd"),
    density_surv = get_item("density_surv"),
    density_repr = get_item("density_repr"),
    n1 = get_item("n1"),
    n1_std = get_item("n1_std"),
    age_wise_n1 = get_item("age_wise_n1"),
    age_wise_n1_std = get_item("age_wise_n1_std"),
    s1 = get_item("s1"),
    fitness = get_item("fitness"),
    entropy = get_item("entropy"),
    junk_death = get_item("junk_death"),
    junk_repr = get_item("junk_repr" ),
    junk_fitness = get_item("junk_fitness")
    )
  return(L)
}

import_data <- function(path, run=1){
  get_record(path, run=1)
  return(get_item_list())
}

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
#    "junk_fitness" : Average fitness as predicted from neutral locus

######################
### PLOT FUNCTIONS ###
######################

# survival and standard deviation (2x1)
# colors represent values for different stages with red being the most recent one, the blue
# line represents junk values for the most recent stage, vertical line is maturation age
# the green line represents standard deviation
plot_survival <- function(dirpath=path){
    png(paste(dirpath, "/figures/surv.png", sep="")) # redirect output (device) to png file

    layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
    plot(1-death_mean[[1]], ylim = c(min(1-L$d_range), max(1-L$d_range)), type="l", col=colors[1], xlab = "", ylab = "survival", main="Survival")
    for(i in 2:16)
      {lines(1-death_mean[[i]], col=colors[i])}
    abline(v=L$m)
    lines(1-rep(L$junk_death[16],71), col="blue") # junk

    plot(death_sd[[16]], type="o", col="green", xlab = "age", ylab = "sd") # sd
    abline(v=L$m) # maturation

    dev.off() # close device
}

# reproduction and standard deviation (2x1)
# colors represent values for different stages with red being the most recent one, the blue
# line represents junk values for the most recent stage, vertical line is maturation age
# the green line represents standard deviation
plot_reproduction <- function(dirpath=path){
    png(paste(dirpath, "/figures/repr.png", sep="")) # redirect output (device) to png file

    layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
    plot(repr_mean[[1]], xlim = c(16, 70), ylim = c(min(L$r_range), max(L$r_range)), type="l", col=colors[1], xlab = "", ylab = "reproduction", main="Reproduction")
    for(i in 2:16)
      {lines(repr_mean[[i]], col=colors[i])}
    abline(v=L$m)
    lines(rep(L$junk_repr[16],71), col="blue") # junk

    plot(repr_sd[[16]], xlim = c(16, 70), type="o", col="green", xlab = "age", ylab = "sd") # sd
    abline(v=L$m)

    dev.off() # close device
}

# population (blue) and resources (red)
plot_pop_res <- function(dirpath=path){
    png(paste(dirpath, "/figures/pop_res.png", sep="")) # redirect output (device) to png file

    plot(L$resources, type="l", col="red", xlab="stage", ylab="N", ylim=c(0,max(L$resources,L$population_size)), main="pop (blue) and res (red)")
    lines(L$population_size, type="l", col="blue")
    lines(L$resources, type="l", col="red") ###

    dev.off() # close device
}

# age distribution
# colors represent values for different stages with red being the most recent one
plot_age_distr <- function(dirpath=path){
    png(paste(dirpath, "/figures/age_dist.png", sep="")) # redirect output (device) to png file

    plot(age_distribution[[1]], type="l", col=colors[[1]], ylim=c(0, max(age_distribution)), xlab = "age", ylab = "", main="Age distribution")
    for(i in 2:16)
      {lines(age_distribution[[i]], col=colors[[i]])}

    dev.off() # close device
}

# frequency of 1's
# (n1 is already sorted in ascending order)
# red lines mark maturation and where reproduction begins [order: survival, reproduction]
plot_frequency <- function(dirpath=path, ix=16, all=FALSE){
    png(paste(dirpath, "/figures/n1.png", sep="")) # redirect output (device) to png file

    if(all){
      par(mar=c(1,1,1,1))
      layout(matrix(1:16,4,4), 1, 1) # 4x4 figure
      plot(n1[[1]], xlab="", ylab="",  ylim=c(0,1), pch=20)
      abline(v=L$m * L$b)
      abline(v=L$maxls * L$b, col="red")
      
      for(i in 2:16){
        plot(n1[[i]],xlab="", ylab="", ylim=c(0,1), pch=20)
        abline(v=L$m * L$b)
        abline(v=L$maxls * L$b, col="red")
      }
    } else{
      layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
      plot(n1[[ix]], xlab="position", ylab="frequency", main="Frequency of 1's", ylim=c(0,1))
      abline(v=L$m * L$b)
      abline(v=L$maxls * L$b, col="red")
      plot(n1_std[[ix]], xlab="position", ylab="sd", type="l")
      abline(v=L$m * L$b)
      abline(v=L$maxls * L$b, col="red")
    }
    dev.off() # close device
}

# age wise frequency of 1's
plot_age_wise_var <- function(dirpath=path, ix=16, all=FALSE){
    png(paste(dirpath, "/figures/age_wise_n1.png", sep="")) # redirect output (device) to png file

    if(all){
        par(mar=c(1,1,1,1))
        layout(matrix(1:16,4,4), 1, 1) # 4x4 figure
        plot(age_wise_n1[[1]], xlab="", ylab="",  ylim=c(0,1), pch=20)
        abline(v=L$m)
        abline(v=L$maxls, col="red")

        for(i in 2:16){
          plot(age_wise_n1[[i]],xlab="", ylab="", ylim=c(0,1), pch=20)
          abline(v=L$m)
          abline(v=L$maxls, col="red")
        }
    } else{
        layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
        plot(age_wise_n1[[ix]], xlab="age", ylab="frequency", main="Age-wise frequency of 1's", ylim=c(0,1))
        abline(v=L$m)
        abline(v=L$maxls, col="red")
        plot(age_wise_n1_std[[ix]], xlab="age", ylab="sd")
        abline(v=L$m)
        abline(v=L$maxls, col="red")
  }
  dev.off()
}

# density of genotypes (age unspecific)
# survival (green) and reproduction (red)
plot_density <- function(dirpath=path, ix=16, all=FALSE){
    png(paste(dirpath, "/figures/dens.png", sep="")) # redirect output (device) to png file

    if(all){
      layout(matrix(1:16,4,4), 1, 1) # 4x4 figure
      plot(density_repr[[1]], type="l", col="red", xlab="Genotype", ylab="", ylim=c(0, max(density_repr, density_surv)))
      lines(density_surv[[1]], col="green")
      
      for(i in 2:16){
        plot(density_repr[[i]], type="l", col="red", xlab="Genotype", ylab="", ylim=c(0, max(density_repr, density_surv)))
        lines(density_surv[[i]], col="green")
      }
    } else{
      plot(density_repr[[ix]], type="l", col="red", xlab="Genotype", ylab="density", ylim=c(0, max(density_repr[[ix]], density_surv[[ix]])), main="Genotype (non age-specific) density")
      lines(density_surv[[ix]], col="green")
    }
    dev.off() # close device
}

# actual death rate (calculated from population_size * age_distribution)
# averaged over s1:s2
plot_actual_death_rate <- function(dirpath=path, s1, s2){
    png(paste(path, "/figures/actual_death_rate.png", sep="")) # redirect output (device) to png file

    plot(rowMeans(cbind(actual_death_rate[c(s1:s2)])),xlab="age", ylab=expression(mu), main="Actual death rate")
    dev.off() # close device
}

# entropy
plot_entropy <- function(dirpath=path){
    png(paste(dirpath, "/figures/entropy.png", sep="")) # redirect output (device) to png file

    plot(L$entropy, type="o", main="Entropy")
    dev.off() # close device
}

# plot all
plot_all <- function(dirpath=path, ix=16, all=FALSE, s1=n_stages-101, s2=n_stages-1){
  plot_survival(path)
  plot_reproduction(path)
  plot_pop_res(path)
  plot_age_distr(path)
  plot_frequency(path, ix, all)
  plot_age_wise_var(path, ix, all)
  plot_density(path, ix, all)
  plot_actual_death_rate(path, s1, s2)
  plot_entropy(path)
}

###############
### EXECUTE ###
###############

path <- ""
L = import_data(path)

# reshape data frames
age_distribution = data.frame(t(L$age_distribution))[L$snapshot_stages]
death_mean = data.frame(t(L$death_mean))
death_sd = data.frame(t(L$death_sd))
actual_death_rate = data.frame(t(L$actual_death_rate))
repr_mean = data.frame(t(L$repr_mean))
repr_sd = data.frame(t(L$repr_sd))
density_surv = data.frame(t(L$density_surv))
density_repr = data.frame(t(L$density_repr))
n1 = data.frame(t(L$n1))
n1_std = data.frame(t(L$n1_std))
age_wise_n1 = data.frame(t(L$age_wise_n1))
age_wise_n1_std = data.frame(t(L$age_wise_n1_std))
#s1 =
#fitness = data.frame(t(L$fitness))

# plot values
colors <- rev(heat.colors(16)) # color palette
n_stages <- length(L$population_size)

plot_all(path, ix=16, all=FALSE, s1=n_stages-101, s2=n_stages-1)
