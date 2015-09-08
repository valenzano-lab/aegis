def x_bincount(array):
    return np.bincount(array, minlength=3)

def some_report_function(population, N, gen_map, chr_len, n_bases, 
        d_range, r_range, max_ls):
    # To save: mean and sd survival and reproduction
    # Distribution of genotypes
    # population fitness
    # Average 1's per locus
    # Sliding-window SD of ^
    # N
    # Resources

    # Determine if population is already sorted; if not, sort it by age:
    
    ages = population[:,0]
    if ages != ages[ages.argsort()]:
        population = population[ages.argsort()]
    b = n_bases # Binary units per locus

    ## AGE-DEPENDENT STATS ##
    # Genotype sum distributions:
    density_surv = np.zeros((2*b+1,))
    density_repr = np.zeros((2*b+1,))
    # Mean death/repr rates by age:
    death_rate_out = np.zeros(max_ls)
    repr_rate_out = np.zeros(max_ls)
    # Death/repr rate SD by age:
    surv_rate_sd = np.zeros(max_ls)
    repr_rate_sd = np.zeros(max_ls)
    # Mean fitness (normalised survival * reproduction) by age
    fitness = np.zeros(max_ls)
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
        death_rates = d_range[surv_gen]
        # Calculate statistics:
        death_rate_out[x] = np.mean(death_rates)
        death_rate_sd[x] = np.std(death_rates)
        density_surv += np.bincount(surv_gen, minlength=2*b+1)
        if x>=maturity:
            # Same for reproduction if they're adults
            repr_locus = np.nonzero(gen_map==(age+100))[0][0]
            repr_pos = np.arange(repr_locus*b, (repr_locus+1)*b)+1
            repr_pop = pop[np.append(repr_pos, repr_pos+chr_len)]
            repr_gen = np.sum(repr_pop, axis=0)
            repr_rates = r_range[repr_gen]
            repr_rate_out[x] = np.mean(repr_rates)
            repr_rate_sd[x] = np.std(repr_rates)
            density_repr += np.bincount(repr_gen, minlength=2*b+1)
            fitness[x] = np.mean(surv_gen/20 * repr_gen/20)

    ## AGE-INVARIANT STATS ##
    # Shannon-Weaver index as measure of heterozygosity at each position:
    chr1 = population[:,range(chr_len)+1]
    chr2 = population[:,range(chr_len)+1+chr_len]
    hetz = np.apply_along_axis(x_bincount, 0, chr1+chr2)/N
    sh_w = np.apply_along_axis(scipy.stats.entropy, 0, hetz)
    # Junk stats calculated from neutral locus
    neut_locus = np.nonzero(gen_map==201)[0][0] 
    neut_pos = np.arange(neut_locus*b, (neut_locus+1)*b)+1
    neut_pop = population[np.append(neut_pos, neut_pos+chr_len)]
    neut_gen = np.sum(neut_pop, axis=1)
    death_rate_junk_out = np.mean(d_range[neut_gen])
    repr_rate_junk_out = np.mean(r_range[neut_gen]) # Junk SDs?
    fitness_junk = np.mean((neut_gen/20)**2)

            ## average the output data 
            density = (density_surv+density_repr)/(126*len(population))
            density_surv = density_surv/(71*len(population))
            density_repr = density_repr/(55*len(population))
            hetrz_mea = hetrz_mea/(2*len(population))

            ## standard deviation
            for i in range(71):
                surv_rate_sd[i] = np.sqrt(np.mean(np.square(surv_rate_sd[i]-surv_rate_out[i])))
                repr_rate_sd[i] = np.sqrt(np.mean(np.square(repr_rate_sd[i]-repr_rate_out[i])))
            
            for i in range(1260):
                hetrz_mea_sd[i] = np.sqrt(np.mean(np.square(hetrz_mea_sd[i]-hetrz_mea[i])))

            ## append to text file
            repr_rate_txt.append(repr_rate_out)
            repr_rate_sd_txt.append(repr_rate_sd) # sd
            repr_rate_junk_txt.append(repr_rate_junk_out)
            surv_rate_txt.append(surv_rate_out)
            surv_rate_sd_txt.append(surv_rate_sd) # sd
            surv_rate_junk_txt.append(surv_rate_junk_out)
            repr_fit_txt.append(repr_fit)
            repr_fit_junk_txt.append(repr_fit_junk)
            surv_fit_txt.append(surv_fit)
            surv_fit_junk_txt.append(surv_fit_junk)
            density_surv_txt.append(density_surv)
            density_repr_txt.append(density_repr)
            hetrz_mea_txt.append(hetrz_mea)
            hetrz_mea_sd_txt.append(hetrz_mea_sd)

