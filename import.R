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

### FULL LIST OF RECORD ITEMS ###
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
