### read csv files ###
path <- "~/repos/genome-simulation/test/"

# shape = (71, 16) [per-age data]
age_distribution <- read.csv(paste(path, "csv_output/age_distribution.csv", sep=""), header=FALSE)
death_mean <- read.csv(paste(path, "csv_output/death_mean.csv", sep=""), header=FALSE)
repr_sd <- read.csv(paste(path, "csv_output/repr_sd.csv", sep=""), header=FALSE)
fitness <- read.csv(paste(path, "csv_output/fitness.csv", sep=""), header=FALSE)
death_sd <- read.csv(paste(path, "csv_output/death_sd.csv", sep=""), header=FALSE)
repr_mean <- read.csv(paste(path, "csv_output/repr_mean.csv", sep=""), header=FALSE)

# shape = (, n_stage) [per-stage data]
resources <- read.csv(paste(path, "csv_output/resources.csv", sep=""), header=FALSE)[[1]]
population_size <- read.csv(paste(path, "csv_output/population_size.csv", sep=""), header=FALSE)[[1]]
starvation_factor <- read.csv(paste(path, "csv_output/starvation_factor.csv", sep=""), header=FALSE)[[1]]
#
actual_death_rate <- read.csv(paste(path, "csv_output/actual_death_rate.csv", sep=""), header=FALSE)

# shape = (21, 16) [genotype data]
density_surv <- read.csv(paste(path, "csv_output/density_surv.csv", sep=""), header=FALSE) / 100
density_repr <- read.csv(paste(path, "csv_output/density_repr.csv", sep=""), header=FALSE) / 100

n1 <- read.csv(paste(path, "csv_output/n1.csv", sep=""), header=FALSE)
n1_std <- read.csv(paste(path, "csv_output/n1_sd.csv", sep=""), header=FALSE)
age_wise_n1_std <- read.csv(paste(path, "csv_output/age_wise_n1_sd.csv", sep=""), header=FALSE)
#s1 <- read.csv(paste(path, "csv_output/s1.csv", sep=""), header=FALSE)

# per-snapshot data
entropy <- (read.csv(paste(path, "csv_output/entropy.csv", sep=""), header=FALSE))[[1]]
junk_death <- (read.csv(paste(path, "csv_output/junk_death.csv", sep=""), header=FALSE))[[1]]
junk_repr <- (read.csv(paste(path, "csv_output/junk_repr.csv", sep=""), header=FALSE))[[1]]
junk_fitness <- (read.csv(paste(path, "csv_output/junk_fitness.csv", sep=""), header=FALSE))[[1]]

# population parameters and other
# input as vectors, not matrices
gen_map <- (read.csv(paste(path, "csv_output/gen_map.csv", sep=""), header=FALSE))[[1]]
chr_len <- (read.csv(paste(path, "csv_output/chr_len.csv", sep=""), header=FALSE))[[1]]
n_bases <- (read.csv(paste(path, "csv_output/n_bases.csv", sep=""), header=FALSE))[[1]]
max_ls <- (read.csv(paste(path, "csv_output/max_ls.csv", sep=""), header=FALSE))[[1]]
maturity <- (read.csv(paste(path, "csv_output/maturity.csv", sep=""), header=FALSE))[[1]]
d_range <- (read.csv(paste(path, "csv_output/d_range.csv", sep=""), header=FALSE))[[1]]
r_range <- (read.csv(paste(path, "csv_output/r_range.csv", sep=""), header=FALSE))[[1]]
snapshot_stages <- (read.csv(paste(path, "csv_output/snapshot_stages.csv", sep=""), header=FALSE))[[1]]

######################
### PLOT FUNCTIONS ###
######################

# plot values
colors <- rev(heat.colors(16)) # color palette
n_stages <- length(population_size)

# survival and standard deviation (2x1)
# colors represent values for different stages with red being the most recent one, the blue
# line represents junk values for the most recent stage, vertical line is maturation age
# the green line represents sd
plot_survival <- function(dirpath=path){
png(paste(dirpath, "figures/surv.png", sep="")) # redirect output (device) to png file

layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
plot(1-death_mean[[1]], ylim = c(0.98, 1), type="l", col=colors[1], xlab = "", ylab = "survival", main="Survival")
for(i in 2:16)
  {lines(1-death_mean[[i]], col=colors[i])}
abline(v=16) # maturation
lines(1-rep(junk_death[16],71), col="blue") # junk

plot(death_sd$V16, type="o", col="green", xlab = "age", ylab = "sd") # sd
abline(v=16) # maturation

dev.off() # close device
}

# reproduction and standard deviation (2x1)
# colors represent values for different stages with red being the most recent one, the blue
# line represents junk values for the most recent stage, vertical line is maturation age
# the green line represents sd
plot_reproduction <- function(dirpath=path){
png(paste(dirpath, "figures/repr.png", sep="")) # redirect output (device) to png file

layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
plot(repr_mean[[1]], xlim = c(16, 70), ylim = c(min(r_range), max(r_range)), type="l", col=colors[1], xlab = "", ylab = "reproduction", main="Reproduction")
for(i in 2:16)
  {lines(repr_mean[[i]], col=colors[i])}
abline(v=16) # maturation
lines(rep(junk_repr[16],71), col="blue") # junk

plot(repr_sd$V16, xlim = c(16, 70), type="o", col="green", xlab = "age", ylab = "sd") # sd
abline(v=16)

dev.off() # close device
}

# population (blue) and resources (red), and starvation factor m
plot_pop_res <- function(dirpath=path){
png(paste(dirpath, "figures/pop_res.png", sep="")) # redirect output (device) to png file

layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
plot(resources, type="l", col="red", xlab="", ylab="", main="Population")
lines(population_size, type="l", col="blue")
plot(starvation_factor, xlab="stage", ylab="m")

dev.off() # close device
}

# age distribution
# colors represent values for different stages with red being the most recent one
plot_age_distr <- function(dirpath=path){
png(paste(dirpath, "figures/age_dist.png", sep="")) # redirect output (device) to png file

plot(age_distribution[[1]], type="l", col=colors[[1]], ylim=c(0, max(age_distribution)), xlab = "", ylab = "", main="Age distribution")
for(i in 2:16)
  {lines(age_distribution[[i]], col=colors[[i]])}

dev.off() # close device
}

# frequency of 1's
# sorted by rec_to_csv.py
# red lines represent maturation and where reproduction begins [order: survival, reproduction]
plot_frequency <- function(dirpath=path, ix=16, all=FALSE){
png(paste(dirpath, "figures/n1.png", sep="")) # redirect output (device) to png file

if(all){
  par(mar=c(1,1,1,1))
  layout(matrix(1:16,4,4), 1, 1) # 4x4 figure
  plot(n1[[1]], xlab="", ylab="",  ylim=c(0,1), pch=20)
  abline(v=160, col="red")
  abline(v=710, col="red")
  
  for(i in 2:16){
    plot(n1[[i]],xlab="", ylab="", ylim=c(0,1), pch=20)
    abline(v=160, col="red")
    abline(v=710, col="red")
  }
} else{
  layout(matrix(1:2,2,1), 2, 1) # 2x1 figure
  plot(n1[[ix]], xlab="position", ylab="frequency", main="Frequency of 1's")
  abline(v=160, col="red")
  abline(v=710, col="red")
  plot(n1_std[[ix]], xlab="position", ylab="std", type="l")
  abline(v=160, col="red")
  abline(v=710, col="red")
}
dev.off() # close device
}

# age wise standard deviation od frequency of 1's
plot_age_wise_var <- function(dirpath=path, ix=16, all=FALSE){
  png(paste(dirpath, "figures/n1.png", sep="")) # redirect output (device) to png file

  if(all){
    par(mar=c(1,1,1,1))
    layout(matrix(1:16,4,4), 1, 1) # 4x4 figure
    plot(age_wise_n1_std[[1]], xlab="", ylab="",  ylim=c(0,1), pch=20)
    abline(v=16, col="red")
    abline(v=71, col="red")
    
    for(i in 2:16){
      plot(age_wise_n1_std[[i]],xlab="", ylab="", ylim=c(0,1), pch=20)
      abline(v=16, col="red")
      abline(v=71, col="red")
    }
  } else{
    plot(age_wise_std_n1[[ix]], xlab="position", ylab="frequency", main="Age-wise genome variation")
    abline(v=16, col="red")
    abline(v=71, col="red")
  }
  dev.off()
}
# density of genotypes (displayed age unspecific)
# survival (green) and reproduction (red), first (up) and most recent stage (down)
plot_density <- function(dirpath=path, ix=16, all=FALSE){
png(paste(dirpath, "figures/dens.png", sep="")) # redirect output (device) to png file

if(all){
  layout(matrix(1:16,4,4), 1, 1) # 4x4 figure
  plot(density_repr[[1]], type="l", col="red", xlab="", ylab="", ylim=c(0, max(density_repr, density_surv)))
  lines(density_surv[[1]], col="green")
  
  for(i in 2:16){
    plot(density_repr[[i]], type="l", col="red", xlab="", ylab="", ylim=c(0, max(density_repr, density_surv)))
    lines(density_surv[[i]], col="green")
  }
} else{
  plot(density_repr[[ix]], type="l", col="red", xlab="", ylab="density", ylim=c(0, max(density_repr[[ix]], density_surv[[ix]])), main="Genotype (non age-specific) density")
  lines(density_surv[[ix]], col="green")
}
dev.off() # close device
}

# actual death rate (calculated from population_size * age_distribution)
# averaged over s1:s2
plot_actual_death_rate <- function(dirpath=path, s1, s2){
png(paste(path, "figures/actual_death_rate.png", sep="")) # redirect output (device) to png file

plot(rowMeans(cbind(actual_death_rate[c(s1:s2)])), ylab=expression(mu), main="Actual death rate")
dev.off() # close device
}
# entropy
plot_entropy <- function(dirpath=path){
png(paste(dirpath, "figures/entropy.png", sep="")) # redirect output (device) to png file

plot(entropy, type="o", main="Entropy")

dev.off() # close device
}

# all
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

### EXECUTED CODE BLOCK ###
plot_all()
