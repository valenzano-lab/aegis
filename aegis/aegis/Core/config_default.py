################################################
## GENOME SIMULATION v.2.1 CONFIGURATION FILE ##
################################################

## CORE PARAMETERS ##

number_of_runs = 2 # Total number of independent runs
number_of_stages = 20 # Total number of stages per run
number_of_snapshots = 10 # Points in run at which to record detailed data
repr_mode = 'sexual' # sexual, asexual, assort_only or recombine_only
res_start = 1000 # Starting resource value
res_var = True # Resources vary with population and time; else constant
output_mode = 1 # 0 = return records only, 1 = return records + final pop,
                # 2 = return records + all snapshot populations

## RESOURCE PARAMETERS ##
V = 1.1 # Geometric resource regrowth factor, if variable
R = res_start # Arithmetic resource increment, if variable
res_limit = res_start*5 # Maximum resource value, if variable; -1 = infinite

## STARTING POPULATION PARAMETERS ##
start_pop = res_start # Starting population size
g_dist_s = 0.5 # Proportion of 1's in survival loci of initial genomes
g_dist_r = g_dist_s #                       reproductive loci
g_dist_n = g_dist_s #                       neutral loci

## SIMULATION FUNDAMENTALS: CHANGE WITH CARE ##
death_bound = [0.001, 0.02] # min and max death rates
repr_bound = [0, 0.2] # min and max reproduction rates
r_rate = 0.01 # recombination rate, if sexual
m_rate = 0.001 # mutation rate
m_ratio = 0.1 # Ratio of positive (0->1) to negative (1->0) mutations
repr_offset = 100 # Offset for repr loci in genome map (must be <= max_ls)
neut_offset = 200 # Offset for neut loci (<= repr_offset + max_ls - maturity)
max_ls = 98 # Maximum lifespan (must be > repr_offset) (-1 = infinite)
maturity = 21 # Age from which an individual can reproduce (must be <= max_ls)
n_neutral = 10 # Number of neutral loci in genome
n_base = 10 # Number of bits per locus
surv_pen = True # Survival penalty under starvation
repr_pen = False # Reproduction penalty under starvation
death_inc = 3 # Per-stage death rate increase under starvation
repr_dec = death_inc # Per-stage reproduction rate decrease under starvation
# Size of sliding windows for recording averaged statistics:
windows = {"population_size": 1000, "resources":1000, "n1":10}
