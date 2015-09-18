################################################
## GENOME SIMULATION v.1.5 CONFIGURATION FILE ##
################################################

sexual = False # Sexual or asexual reproduction

## RUNNING PARAMETERS ## 
number_of_runs = 1 # Total number of independent runs
number_of_stages = 200 # Total number of stages per run
crisis_stages = [] # Stages of extrinsic death crisis
crisis_sv = 0.05 # Fraction of crisis survivors (if applicable)
number_of_snapshots = 16 # Number of stages at which to store output data

## RESOURCE PARAMETERS ##
res_start = 0 # Starting resource value
res_var = True # Resources vary with population and time; else constant
R = 1000 # Per-stage resource increment, if variable
V = 1.6 # Resource regrowth factor, if variable
res_limit = 5000 # Maximum resource value, if variable

## STARTING POPULATION PARAMETERS ##
start_pop = 500 # Starting population size
age_random = True # Random starting ages; else all start as new adults
# Proportion of 1's and 0's in the genome can be randomly chosen for each 
# chromosome in each individual (drawn from a truncated normal about 0.5) or 
# set at a constant percentage value, separately for survival and reproductive 
# loci:
s_dist = "random" # "random" or PROPORTIONAL value (0.0-1.0) for S loci
r_dist = "random" # "random" or PROPORTIONAL value (0.0-1.0) for R loci
variance = 1.4 # Variance of parameter distribution, if random

## SIMULATION FUNDAMENTALS: CHANGE WITH CARE ##
death_bound = [0.001, 0.02] # min and max death rates
repr_bound = [0, 0.2] # min and max reproduction rates
r_rate = 0.01 # recombination rate, if sexual
m_rate = 0.001 # mutation rate
m_ratio = 0.1 # Ratio of positive (0->1) to negative (1->0) mutations
max_ls = 71 # Maximum lifespan; must be less than 100
maturity = 16 # Age of sexual maturation
n_base = 10 # Genome size (binary units per locus)
death_inc = 3 # Per-stage death-rate increase under starvation
window_size = 10 # Size of sliding window for recording p1 SD

## DERIVED PARAMETERS ##
import numpy as np
gen_map = np.asarray(range(0,max_ls)+range(maturity+100,
    max_ls+100)+[201])
# Genome map: survival (0 to max), reproduction (maturity to max), neutral
chr_len = len(gen_map)*n_base # Length of chromosome in binary units
d_range = np.linspace(death_bound[1], death_bound[0],2*n_base+1) 
# max to min death rate
r_range = np.linspace(repr_bound[0],repr_bound[1],2*n_base+1) 
# min to max repr rate
snapshot_stages = int(np.around(np.linspace(0,number_of_stages-1,
    number_of_snapshots),0)) # Stages to save detailed record
