################################################
## GENOME SIMULATION v.2.1 CONFIGURATION FILE ##
################################################

sexual = False # Sexual or asexual reproduction

## RUNNING PARAMETERS ##
number_of_runs = 1 # Total number of independent runs
number_of_stages = 0 # Total number of stages per run
crisis_stages = [] # Stages of extrinsic death crisis
crisis_sv = 0.05 # Fraction of crisis survivors (if applicable)
number_of_snapshots = 16 # Number of stages at which to store output data;
                         # if integer, will take that many snapshots
                         # evenly distributed throughout stages; if float,
                         # will take a snapshot at that proportion of stages

## RESOURCE PARAMETERS ##
res_start = 0 # Starting resource value
res_var = True # Resources vary with population and time; else constant
R = 1000 # Per-stage resource increment, if variable
V = 1.6 # Resource regrowth factor, if variable
res_limit = 5000 # Maximum resource value, if variable

## STARTING POPULATION PARAMETERS ##
start_pop = 500 # Starting population size
age_random = False # Random starting ages; else all start as new adults
g_dist_s = 0.3 # Propoprtion of 1's in survival loci of initial genomes
g_dist_r = 0.8 #                       reproductive loci
g_dist_n = 0.2 #                       neutral loci

## SIMULATION FUNDAMENTALS: CHANGE WITH CARE ##
death_bound = [0.001, 0.02] # min and max death rates
repr_bound = [0, 0.2] # min and max reproduction rates
r_rate = 0.01 # recombination rate, if sexual
m_rate = 0.001 # mutation rate
m_ratio = 0.1 # Ratio of positive (0->1) to negative (1->0) mutations
max_ls = 71 # Maximum lifespan; must be less than 100
maturity = 16 # Age of sexual maturation
n_neutral = 1 # Number of neutral loci in genome
n_base = 10 # Genome size (binary units per locus)
surv_pen = True # Survival penalty under starvation
repr_pen = False # Reproduction penalty under starvation
death_inc = 3 # Per-stage death rate increase under starvation
repr_dec = 3 # Per-stage reproduction rate decrease under starvation
window_size = 10 # Size of sliding window for recording standard deviation of locus
                 # genotypes along genomes.
