import numpy as np
import cPickle
import os

# run from cwd

# load the record file
f = open("./run_1_rec.txt")
record = cPickle.load(f)
f.close()

# check if dir exists and create it if necessary
if not os.path.exists("csv_output"):
    os.makedirs("csv_output")

# compute actual death rate
def actual_death_rate(age_array):

age_dist = record["age_distribution"]
N_age = age_dist * np.tile(record["population_size"].reshape(age_dist.shape[0], 1), age_dist.shape[1])
dividend = N_age[1:, 1:]
divisor = N_age[:-1, :-1]
divisor[divisor == 0] = 1 # avoid deviding with zero
# death rate for last age is 1
actual_death_rate = np.append((1 - dividend / divisor), np.ones((dividend.shape[0], 1)), axis=1)

### OUTPUT ###
# shape = (16, 71) [per-age data]
    # transpose so that R reads rows-columns correctly
np.savetxt("csv_output/death_mean.csv", record["death_mean"].T, delimiter=",")
np.savetxt("csv_output/repr_sd.csv", record["repr_sd"].T, delimiter=",")
np.savetxt("csv_output/fitness.csv", record["fitness"].T, delimiter=",")
np.savetxt("csv_output/death_sd.csv", record["death_sd"].T, delimiter=",")
np.savetxt("csv_output/repr_mean.csv", record["repr_mean"].T, delimiter=",")

# shape = (n_stage,) [per-stage data]
np.savetxt("csv_output/resources.csv", record["resources"], delimiter=",")
np.savetxt("csv_output/population_size.csv", record["population_size"], delimiter=",")
np.savetxt("csv_output/starvation_factor.csv", record["starvation_factor"], delimiter=",")
# output only snapshot stages
np.savetxt("csv_output/age_distribution.csv", (record["age_distribution"][record["snapshot_stages"].astype(int)-1]).T, delimiter=",")
#
np.savetxt("csv_output/actual_death_rate.csv", actual_death_rate.T, delimiter=",")

# shape = (16, 21) [genotype data]
np.savetxt("csv_output/density_surv.csv", record["density_surv"].T, delimiter=",")
np.savetxt("csv_output/density_repr.csv", record["density_repr"].T, delimiter=",")

# frequency of 1's
# sort before saving: survival, reproduction, junk
def sort_n1(key="n1"):
    b = record["n_bases"]
    m = record["maturity"]
    count = 0
    n1_sort = np.zeros(record[key].shape)
    for i in record["gen_map"]:
        if i<100: # survival
            n1_sort[:, range(i*b, (i+1)*b)] = record[key][:, range(count, count+10)]
        elif i==201: # junk
            n1_sort[:, -10:] = record[key][:, range(count, count+10)]
        else: # reproduction
            n1_sort[:, range(710+(i-100-m)*b, 710+(i+1-100-m)*b)] = record[key][:, range(count, count+10)]
        count += 10
    return n1_sort

np.savetxt("csv_output/n1.csv", sort_n1("n1").T, delimiter=",")
np.savetxt("csv_output/n1_sd.csv", sort_n1("n1_std").T, delimiter=",")
#np.savetxt("csv_output/s1.csv", record["s1"].T, delimiter=",") # rolling n1 sd

# per-snapshot data
np.savetxt("csv_output/entropy.csv", record["entropy"], delimiter=",")
np.savetxt("csv_output/junk_death.csv", record["junk_death"], delimiter=",")
np.savetxt("csv_output/junk_repr.csv", record["junk_repr"], delimiter=",")
np.savetxt("csv_output/junk_fitness.csv", record["junk_fitness"], delimiter=",")

# population paramteres and other
np.savetxt("csv_output/gen_map.csv", record["gen_map"], delimiter=",")
np.savetxt("csv_output/chr_len.csv", record["chr_len"], delimiter=",")
np.savetxt("csv_output/n_bases.csv", record["n_bases"], delimiter=",")
np.savetxt("csv_output/max_ls.csv", record["max_ls"], delimiter=",")
np.savetxt("csv_output/maturity.csv", record["maturity"], delimiter=",")
np.savetxt("csv_output/d_range.csv", record["d_range"], delimiter=",")
np.savetxt("csv_output/r_range.csv", record["r_range"], delimiter=",")
np.savetxt("csv_output/snapshot_stages.csv", record["snapshot_stages"], delimiter=",")
