def some_report_function(population, N, gen_map, chr_len, n_bases, 
        d_range, r_range, max_ls):
    # Determine if population is already sorted; if not, sort it by age:
    ages = population[:,0]
    if ages != ages[ages.argsort()]:
        population = population[ages.argsort()]
    b = n_bases # Binary units per locus

    ## INITIALISE OUTPUT VARIABLES ##

    # Genotype sum distributions:
    density_surv = np.zeros((21,))
    density_repr = np.zeros((21,))
    # Mean death/repr rates by age:
    death_rate_out = np.zeros(max_ls)
    repr_rate_out = np.zeros(max_ls)
    # Death/repr rate SD by age:
    surv_rate_sd = np.zeros(max_ls)
    repr_rate_sd = np.zeros(max_ls)
    # "Fit"?
    surv_fit = np.zeros((71,))
    repr_fit = np.zeros((71,))
    # "Junk"?
    repr_rate_junk_out = np.zeros((1,))
    death_rate_junk_out = np.zeros((1,))
    repr_fit_junk = np.zeros((1,))
    surv_fit_junk = np.zeros((1,))
    # Hetrz?
    hetrz_mea = np.zeros((1260,)) # heterozigosity measure
    hetrz_mea_sd = [[]]*1260 # heterozigosity measure sd

    neut_locus = np.nonzero(gen_map==201)[0][0] 
    neut_pos = np.arange(neut_locus*b, (neut_locus+1)*b)+1
    for x in range(max(ages)):
        pop = population[population[,0]==x]
        # Find loci and binary units:
        surv_locus = np.nonzero(gen_map==age)[0][0]
        surv_pos = np.arange(surv_locus*b, (surv_locus+1)*b)+1
        # Subset array to relevant columns and find genotypes:
        surv_pop = pop[np.append(surv_pos, surv_pos+chr_len)]
        surv_gen = np.sum(surv_pop, axis=0)
        # Find death/reproduction rates:
        death_rates = d_range[surv_gen]
        # Calculate statistics:
        death_rate_out[x] = np.mean(death_rates)
        death_rate_sd[x] = np.std(death_rates)
        surv_fit[x] = ??
        if x>=maturity:
            # Same for reproduction if they're adults
            repr_locus = np.nonzero(gen_map==(age+100))[0][0]
            repr_pos = np.arange(repr_locus*b, (repr_locus+1)*b)+1
            repr_pop = pop[np.append(repr_pos, repr_pos+chr_len)]
            repr_gen = np.sum(repr_pop, axis=0)
            repr_rates = r_range[repr_gen]
            repr_rate_out[x] = np.mean(repr_rates)
            repr_rate_sd[x] = np.std(repr_rates)
            repr_fit[x] = ??
        # Junk variables??

            
                # Survival statistics:
                surv_rate_sd[surv_locus].append(1-death_rate_var[surv_out])
                surv_fit[surv_locus] += surv_fit_var[surv_out]
                for t in range(surv_pos[0], surv_pos[1]):
                    hetrz = I[1][t]+I[2][t]
                    hetrz_mea[t] += hetrz
                    hetrz_mea_sd[t].append(hetrz)
                # Reproduction statistics:
                repr_rate_out[repr_locus] += repr_rate_var[repr_out]
                repr_rate_sd[repr_locus].append(repr_rate_var[repr_out])
                repr_fit[repr_locus] += repr_fit_var[repr_out]
                for t in range(repr_pos[0], repr_pos[1]):
                    hetrz = I[1][t]+I[2][t]
                    hetrz_mea[t] += hetrz
                    hetrz_mea_sd[t].append(hetrz)
                # Neutral statistics
                death_rate_junk_out[0] += death_rate_var[neut_out]
                surv_fit_junk[0] += surv_fit_var[neut_out]
                repr_rate_junk_out[0]+=repr_rate_var[neut_out]
                repr_fit_junk[0]+=repr_fit_var[neut_out]

            ## average the output data 
            surv_rate_out = 1-death_rate_out/len(population)
            surv_rate_junk_out = 1-death_rate_junk_out/len(population)
            repr_rate_out = repr_rate_out/len(population)
            repr_rate_junk_out = repr_rate_junk_out/len(population)
            surv_fit = surv_fit/len(population)
            surv_fit_junk = surv_fit_junk/len(population)
            repr_fit = repr_fit/len(population)
            repr_fit_junk = repr_fit_junk/len(population)
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

