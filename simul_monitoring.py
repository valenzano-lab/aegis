        if n_stage in np.linspace(0,number_of_stages,16).astype(int):
            ## reseting output variables
            repr_rate_out = np.zeros((71,))
            repr_rate_junk_out = np.zeros((1,))
            death_rate_out = np.zeros((71,))
            death_rate_junk_out = np.zeros((1,))
            # sd-collect genome values for every age
            repr_rate_sd = [[]]*71
            surv_rate_sd = [[]]*71
            repr_fit = np.zeros((71,))
            repr_fit_junk = np.zeros((1,))
            surv_fit = np.zeros((71,))
            surv_fit_junk = np.zeros((1,))
            density_surv = np.zeros((21,))
            density_repr = np.zeros((21,))
            hetrz_mea = np.zeros((1260,)) # heterozigosity measure
            hetrz_mea_sd = [[]]*1260 # heterozigosity measure sd
            
            neut_locus = np.nonzero(gen_map==201)[0][0] # Position of neutral locus on gen_map
            neut_pos = (neut_locus*10, (neut_locus+1)*10) # Positions in genome array corresponding to neutral locus
            for i in range(len(population)):
                I = population[i] 
                surv_locus = np.nonzero(gen_map==I[0])[0][0] # Position of age-appropriate survival locus on gen_map
                repr_locus = np.nonzero(gen_map==(I[0]+100))[0][0] # Position of age-appropriate reproduction locus on gen_map
                surv_pos = (surv_locus*10, (surv_locus+1)*10)  # Positions in genome array corresponding to survival locus
                repr_pos = (repr_locus*10, (repr_locus+1)*10)  # Positions in genome array corresponding to reproduction locus
                # Get genotype sums:
                surv_out = I[1][surv_pos[0]:surv_pos[1]].count(1) + I[2][surv_pos[0]:surv_pos[1]].count(1)  
                repr_out = I[1][repr_pos[0]:repr_pos[1]].count(1) + I[2][repr_pos[0]:repr_pos[1]].count(1)  
                neut_out = I[1][neut_pos[0]:neut_pos[1]].count(1) + I[2][neut_pos[0]:neut_pos[1]].count(1)  
                # Survival statistics:
                density_surv[surv_out] += 1
                death_rate_out[surv_locus] += death_rate_var[surv_out]
                surv_rate_sd[surv_locus].append(1-death_rate_var[surv_out])
                surv_fit[surv_locus] += surv_fit_var[surv_out]
                for t in range(surv_pos[0], surv_pos[1]):
                    hetrz = I[1][t]+I[2][t]
                    hetrz_mea[t] += hetrz
                    hetrz_mea_sd[t].append(hetrz)
                # Reproduction statistics:
                density_repr[repr_out] += 1
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

