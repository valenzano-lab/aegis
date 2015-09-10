def x_bincount(array):
    return np.bincount(array, minlength=3)
def moving_average(a, n):
    c = np.cumsum(a, dtype=float)
    c[n:] = c[n:]-c[:-n]
    return c[(n-1):]/n

def initialise_record(m, max_ls, n_bases, chr_len, window_size):
    array1 = np.zeros([m,max_ls])
    array2 = np.zeros([m,2*n_bases+1])
    array3 = np.zeros(m)
    record = {
    	"surv_mean":copy(array1)
    	"surv_sd":copy(array1)
    	"repr_mean":copy(array1)
    	"repr_sd":copy(array1)
    	"density_surv":copy(array2)
    	"density_repr":copy(array2)
    	"n1":np.zeros([m,chr_len])
    	"s1":np.zeros([m,chr_len-window_size+1])
    	"fitness":copy(array3)
    	"entropy":copy(array3)
    	"surv_junk":copy(array3)
    	"repr_junk":copy(array3)
    	"fitness_junk":copy(array3)
        }
    return record

def update_record(record, population, N, gen_map, chr_len, 
        n_bases, d_range, r_range, max_ls, window_size):

    b = n_bases # Binary units per locus

    ## AGE-DEPENDENT STATS ##
    # Genotype sum distributions:
    density_surv = np.zeros((2*b+1,))
    density_repr = np.zeros((2*b+1,))
    # Mean death/repr rates by age:
    surv_mean = np.zeros(max_ls)
    repr_mean = np.zeros(max_ls)
    # Death/repr rate SD by age:
    surv_sd = np.zeros(max_ls)
    repr_sd = np.zeros(max_ls)
    # Loop over ages:
    for x in range(max(ages)):
        pop = population[population[,0]==x]
        # Find loci and binary units:
        surv_locus = np.nonzero(gen_map==age)[0][0]
        surv_pos = np.arange(surv_locus*b, (surv_locus+1)*b)+1
        # Subset array to relevant columns and find genotypes:
        surv_pop = pop[np.append(surv_pos, surv_pos+chr_len)]
        surv_gen = np.sum(surv_pop, axis=1)
        # Find death/reproduction rates:
        surv_rates = 1-d_range[surv_gen]
        # Calculate statistics:
        surv_mean[x] = np.mean(surv_rates)
        surv_sd[x] = np.std(surv_rates)
        density_surv += np.bincount(surv_gen, minlength=2*b+1)
        if x>=maturity:
            # Same for reproduction if they're adults
            repr_locus = np.nonzero(gen_map==(age+100))[0][0]
            repr_pos = np.arange(repr_locus*b, (repr_locus+1)*b)+1
            repr_pop = pop[np.append(repr_pos, repr_pos+chr_len)]
            repr_gen = np.sum(repr_pop, axis=0)
            repr_rates = r_range[repr_gen]
            repr_mean[x] = np.mean(repr_rates)
            repr_sd[x] = np.std(repr_rates)
            density_repr += np.bincount(repr_gen, minlength=2*b+1)
    # Average densities over whole population
    density_surv /= N
    density_repr /= N
    # Calculate per-age average genomic fitness
    x_surv = np.cumprod(surv_mean)
    fitness = np.cumsum(x_surv * repr_mean)

    ## AGE-INVARIANT STATS ##
    # Frequency of 1's at each position on chromosome:
    # Average over array, then average at each position over chromosomes
    n1s = np.sum(population, 1)/N 
    n1 = (n1s[range(chr_len)+1]+n1s[range(chr_len)+chr_len+1])/2 
    # Standard deviation of 1 frequency over sliding window
    w = window_size
    s1 = sqrt(moving_average(n1**2, w)-moving_average(n1, w)**2)
    # Shannon-Weaver entropy over entire genome population
    gen = population[1:,])
    p1 = np.sum(gen)/np.size(gen)
    entropy = scipy.stats.entropy(np.array([1-p1, p1]))
    # Junk stats calculated from neutral locus
    neut_locus = np.nonzero(gen_map==201)[0][0] 
    neut_pos = np.arange(neut_locus*b, (neut_locus+1)*b)+1
    neut_pop = population[np.append(neut_pos, neut_pos+chr_len)]
    neut_gen = np.sum(neut_pop, axis=1)
    surv_junk = np.mean(1-d_range[neut_gen])
    repr_junk = np.mean(r_range[neut_gen]) # Junk SDs?
    fitness_junk = np.cumsum(np.cumprod(surv_junk) * repr_junk)

    ## APPEND RECORD OBJECT ##
    record["surv_mean"].append(surv_mean)
    record["surv_sd"].append(surv_sd)
    record["repr_mean"].append(repr_mean)
    record["repr_sd"].append(repr_sd)
    record["density_surv"].append(density_surv)
    record["density_repr"].append(density_repr)
    record["fitness"].append(fitness)
    record["n1"].append(n1)
    record["s1"].append(s1)
    record["entropy"].append(entropy)
    record["surv_junk"].append(surv_junk)
    record["repr_junk"].append(repr_junk)
    record["fitness_junk"].append(fitness_junk)

    return record
