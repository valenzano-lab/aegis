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
    gen_map = get_item("gen_map"),
    chr_len = get_item("chr_len"),
    d_range = get_item("d_range"),
    r_range = get_item("r_range"),
    snapshot_stages = get_item("snapshot_stages"),
    population_size = get_item("population_size"),
    resources = get_item("resources"),
    starvation_factor = get_item("starvation_factor"),
    age_distribution = get_item("age_distribution"),
    surv_mean = get_item("surv_mean"),
    surv_sd = get_item("surv_sd"),
    repr_mean = get_item("repr_mean"),
    repr_sd = get_item("repr_sd"),
    density_surv = get_item("density_surv"),
    density_repr = get_item("density_repr"),
    n1 = get_item("n1"),
    s1 = get_item("s1"),
    fitness = get_item("fitness"),
    entropy = get_item("entropy"),
    surv_junk = get_item("surv_junk"),
    repr_junk = get_item("repr_junk" ),
    fitness_junk = get_item("fitness_junk")
    )
  return(L)
}

import_data <- function(path, run=1){
  get_record(path, run=1)
  return(get_item_list())
}

### FULL LIST OF RECORD ITEMS ###
#    "gen_map" : genome map for the run
#    "chr_len" : length of each chromosome in bits
#    "d_range" : range of possible death probabilities, from max to min
#    "r_range" : range of possible reproduction probabilities (min->max)
#    "snapshot_stages" : stages of run at which detailed info recorded
#    "population_size" : Value of N
#    "resources" : Resource level
#    "starvation_factor" : Value of x
#    "age_distribution" : Proportion of population at each age
#    "surv_mean" : Mean genetic survival probability at each age
#    "surv_sd" : SD generic survival probability at each age
#    "repr_mean" : Mean reproductive survival probability at each age
#    "repr_sd" : Mean reproductive survival probability at each age
#    "density_surv" : Distribution of number of 1's at survival loci
#    "density_repr" : Distribution of number of 1's at reproductive loci
#    "n1" : Average number of 1's at each position along the length of the chromosome
#    "s1" : Sliding-window SD of number of 1's along chromosome
#    "fitness" : Average population fitness as predicted from genotypes
#    "entropy" : Shannon-Weaver entropy across entire population array
#    "surv_junk" : Average survival probability as predicted from neutral locus
#    "repr_junk"  : Average reproductive probability as predicted from neutral locus 
#    "fitness_junk" : Average fitness as predicted from neutral locus
