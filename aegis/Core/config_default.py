##############################
## AEGIS CONFIGURATION FILE ##
##############################

## CORE PARAMETERS ##
random_seed = "" # If numeric, sets random seed to that value before execution
n_runs = 1 # Total number of independent runs
n_stages = 1000 # Total number of stages per run [int/"auto"]
n_snapshots = 8 # Points in run at which to record detailed data
path_to_seed_file = "" # Path to simulation seed file, if no seed then ""
    # see README for which parameters are inherited from seed, which are
    # defined anew in this config file
max_fail = 10 # Maximum number of failed attempts tolerated for each run

## OUTPUT SPECIFICATIONS ##
output_prefix = "test" # Prefix for output files within simulation directory
output_mode = 0 # 0 = return records only, 1 = return records + final pop,
                # 2 = return records + all snapshot populations
age_dist_N = "all" # Window size around snapshots stage/generation for no_auto/auto
                   # for which to record age distribution [int/"all"]
                   # "all" saves age distribution at all stages

## STARTING PARAMETERS ##
repr_mode = "asexual" # sexual, asexual, assort_only or recombine_only
res_start = 1000 # Starting resource value
start_pop = res_start # Starting population size

## RESOURCE PARAMETERS ##
res_function = lambda n,r: r # Function for updating resources; here constant
stv_function = lambda n,r: n > r # Function for identifying starvation
kill_at = 0 # stage/generation for no_auto/auto repectively at which to force
            # dieoff, 0 if none

## PENALISATION ##
pen_cuml = True # Is the penalty cumulative? If True the function compounds,
                # otherwise it is always applied on the default value
surv_pen_func = lambda s_range,n,r: 1-(1-s_range)*3 # Survival penalisation function
repr_pen_func = lambda r_range,n,r: r_range # Reproduction penalisation function

## AUTOCOMPUTING GENERATION NUMBER ##
deltabar = 0.01 # Relative error allowed for the deviation from the stationary
                # distribution
scale = 1.01 # Scaling factor applied to target generation estimated for deltabar
max_stages = 500000 # Maximum number of stages to run before terminating

## SIMULATION FUNDAMENTALS: CHANGE WITH CARE ##
surv_bound = [0.98, 0.99] # min and max death rates
repr_bound = [0, 0.5] # min and max reproduction rates
r_rate = 0.01 # recombination rate, if sexual
m_rate = 0.001 # rate of negative mutations
m_ratio = 0.1 # Ratio of positive (0->1) to negative (1->0) mutations
g_dist = {"s": 0.5, # Proportion of 1's in survival loci of initial genomes
        "r": 0.5,   #                      reproductive loci
        "n": 0.5}   #                      neutral loci
n_neutral = 5 # Number of neutral loci in genome
n_base = 5 # Number of bits per locus
repr_offset = 100 # Offset for repr loci in genome map (must be <= max_ls)
neut_offset = 200 # Offset for neut loci (<= repr_offset + max_ls - maturity)
max_ls = 70 # Maximum lifespan (must be > repr_offset) (-1 = infinite)
maturity = 21 # Age from which an individual can reproduce (must be <= max_ls)

# Size of sliding windows for recording averaged statistics:
windows = {"population_size": 1000, "resources":1000, "n1":n_base}
