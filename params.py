##########################################
## GENOME SIMULATION CONFIGURATION FILE ##
##########################################

sexual = False # Sexual or asexual reproduction
verbose = True # Verbose printouts

## RUNNING PARAMETERS ## 
number_of_runs = 1 # Total number of independent runs
number_of_stages = 200 # Total number of stages per run
crisis_stages = [] # Stages of extrinsic death crisis
crisis_sv = 0.05 # Fraction of crisis survivors (if applicable)
comment = 'test'
out = "testrun/" # Output directory

## RESOURCE PARAMETERS ##
res_start = 0 # Starting resource value
res_var = True # Resources vary with population and time; else constant
R = 1000 # Per-stage resource increment, if variable
V = 1.6 # Resource regrowth factor, if variable
res_limit = 5000 # Maximum resource value, if variable

## STARTING POPULATION PARAMETERS ##
start_pop = 500 # Starting population size
age_random = True # Uniform starting age distribution; else all start as new adults
# Proportion of 1's and 0's in the genome can be randomly chosen for each chromosome in each individual (drawn from a truncated normal about 0.5) or set at a constant percentage value, separately for survival and reproductive loci:
s_dist = "random" # "random" or PERCENTAGE value (0-100) for S loci
r_dist = "random" # "random" or PERCENTAGE value (0-100) for R loci
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
