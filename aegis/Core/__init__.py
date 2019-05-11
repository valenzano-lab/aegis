import os, shutil, numpy as np, pandas as pd
from aegis.Core.functions import chance, quantile, fivenum, init_gentimes,\
        init_ages, init_genomes, init_generations, deep_key, deep_eq, make_windows,\
        correct_r_rate, make_mapping, timenow, get_runtime
from aegis.Core.Population import Population
from aegis.Core.Config import Config
from aegis.Core.Record import Record
from aegis.Core.Run import Run
from aegis.Core.Simulation import Simulation
from aegis.Core.Plotter import Plotter

try:
    import cPickle as pickle
except:
    import pickle

###########
### run ###
###########

def run(config_file, report_n, verbose, outpath=""):
    """Execute a complete simulation from a specified config file."""
    s = Simulation(config_file, report_n, verbose, outpath)
    s.execute()
    s.finalise()

###########
### get ###
###########

def getconfig(outpath):
    """Create a default config file at the specified destination."""
    dirpath = os.path.dirname(os.path.realpath(__file__))
    inpath = os.path.join(dirpath, "config_default.py")
    shutil.copyfile(inpath, outpath)

############
### read ###
############

def getrseed(inpath, outpath="", verbose=False):
    """Get prng from a record object."""
    if verbose: starttime = timenow(False)
    if outpath=="": outpath = os.getcwd()
    fin = open(inpath, "r")
    record = pickle.load(fin)
    fin.close()
    outpath = os.path.join(outpath, record["output_prefix"]+".rseed")
    fout = open(outpath, "w")
    pickle.dump(record["random_seed"], fout)
    fout.close()
    if verbose: print get_runtime(starttime, timenow(False), "Runtime")

def getrecinfo(inpath, outpath="", verbose=False):
    """Get information on record object and output a csv."""
    if verbose: starttime = timenow(False)
    if outpath=="": outpath = os.getcwd()
    # load record
    infile = open(inpath)
    rec = pickle.load(infile)
    infile.close()
    outpath = os.path.join(outpath, rec["output_prefix"]+".recinfo")

    def make_df(rec, suffix=""):
        # create dataframe to output
        odf = pd.DataFrame()
        # iterate through record keys and values
        for key,value in rec.items():
            # set initial values for tracked properties: name, type, shape
            name = suffix + "[" + key + "]" if suffix != "" else key
            vtype = [type(value)]
            vshape = [None]
            # if record then recurse call
            if isinstance(value,dict):
                odf = odf.append(make_df(value, suffix=key))
            # if list or tuple set shape to len
            elif isinstance(value,list) or isinstance(value,tuple):
                vshape = [str(len(value))]
            # if numpy array set shape to shape
            elif isinstance(value,np.ndarray):
                vshape = [str(value.shape)]

            df = pd.DataFrame({ "name":name,\
                    "type":vtype,\
                    "shape":vshape})
            odf = odf.append(df)
        # create index and return result
        odf["index"] = np.arange(len(odf))
        odf.set_index("index", inplace=True)
        return odf

    # write to file
    outdf = make_df(rec)
    outdf.to_csv(outpath, index=False)
    if verbose: print get_runtime(starttime, timenow(False), "Runtime")

def get_csv(inpath, outpath="", last_K=500, verbose=False):
    """Only specific data is output from the Record file to a csv files.
    The pandas dataframes are organized with respect to dimensions of the belonging
    numpy arrays in Record."""

    if verbose: starttime = timenow(False)
    # get rec
    infile = open(inpath)
    rec = pickle.load(infile)
    infile.close()

    # keys (some lists are used, some just here to give the overview)

    single_keys = [ "auto",\
                    "dieoff",\
                    "pen_cuml",\
                    "deltabar",\
                    "m_rate",\
                    "m_ratio",\
                    "r_rate",\
                    "r_rate_input",\
                    "scale",\
                    "chr_len",\
                    "kill_at",\
                    "maturity",\
                    "max_fail",\
                    "max_ls",\
                    "max_stages",\
                    "min_gen",\
                    "n_base",\
                    "n_neutral",\
                    "n_snapshots",\
                    "n_stages",\
                    "n_states",\
                    "output_mode",\
                    "res_start",\
                    "start_pop",\
                    "age_dist_N",\
                    "output_prefix",\
                    "path_to_seed_file",\
                    "repr_mode",\
                    "avg_starvation_length",\
                    "avg_growth_length",\
                    # need special treatment since dicts
                    "g_dist_s",\
                    "g_dist_r",\
                    "g_dist_n",\
                    "n1_window_size"]

    nstagex1_keys = [   "poulation_size",\
                        "resources",\
                        # need special treatment since 5 columns
                        "generation_dist",\
                        "gentime_dist"]

    nstagexmaxls_keys = ["age_distribution"]

    maxlsx1_keys = ["kaplan-meier","observed_repr_rate"]

    nsnapxmaxls_keys = ["cmv_surv",\
                        "fitness_term",\
                        "junk_cmv_surv",\
                        "junk_fitness_term",\
                        "junk_repr",\
                        "junk_repr_value",\
                        "mean_repr",\
                        "repr_value",\
                        # need special treatment since dict with surv and repr
                        "prob_mean",\
                        "prob_var"]

    nsnapxnbit_keys = [ "n1",\
                        "n1_var"]

    sliding_window_keys = [ "n1_window_mean",\
                            "n1_window_var"]

    nsnapxnloci_keys = ["mean_gt",\
                        "var_gt"]

    nsnapx1_keys = ["fitness"]

    # construct pandas objects

    # single values
    def single_df(rec, keys):
        df = pd.DataFrame()
        names = keys[:-4]
        values = [rec[key] for key in names]
        for ss in ["s","r","n"]:
            names.append("g_dist_"+ss)
            values.append(rec["g_dist"][ss])
        names.append("n1_window_size")
        values.append(rec["windows"]["n1"])
        df["name"] = names
        df["value"] = values
        return df

    # shape = (nstage,)
    def nstagex1_df(rec):
        df = pd.DataFrame()
        df["stage"] = np.arange(rec["population_size"].size)
        df["popsize"] = rec["population_size"].astype(int)
        df["resources"] = rec["resources"].astype(int)
        # generation distribution
        df["generation_min"] = rec["generation_dist"][:,0]
        df["generation_25_percentile"] = rec["generation_dist"][:,1]
        df["generation_median"] = rec["generation_dist"][:,2]
        df["generation_75_percentile"] = rec["generation_dist"][:,3]
        df["generation_max"] = rec["generation_dist"][:,4]
        # gentime distribution
        df["gentime_min"] = rec["gentime_dist"][:,0]
        df["gentime_25_percentile"] = rec["gentime_dist"][:,1]
        df["gentime_median"] = rec["gentime_dist"][:,2]
        df["gentime_75_percentile"] = rec["gentime_dist"][:,3]
        df["gentime_max"] = rec["gentime_dist"][:,4]
        # bit variance
#        df["bit_variance_premature"] = rec["bit_variance"][:,0]
#        df["bit_variance_mature"] = rec["bit_variance"][:,1]
        return df

    # shape = (nstage, maxls)
    def nstagexmaxls_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["stage"] = np.repeat(np.arange(sh[0]),sh[1])
        df["age"] =  np.tile(np.arange(sh[1]),sh[0])
        for key in keys:
            df[key] = rec[key].flatten()
        return df

    # shape = (maxls,)
    def maxlsx1_df(rec, keys):
        df = pd.DataFrame()
        df["age"] = np.arange(rec[keys[0]].size)
        for key in keys:
            df[key] = rec[key]
        return df

    # shape = (nsnap, maxls)
    def nsnapxmaxls_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["age"] =  np.tile(np.arange(sh[1]),sh[0])
        for key in keys[:-2]:
            df[key] = rec[key].flatten()
        for ss in ["mean", "var"]:
            df["surv_prob_"+ss] = rec["prob_"+ss]["surv"].flatten()
            df["repr_prob_"+ss] = np.concatenate((np.zeros((sh[0],rec["maturity"])),\
                                    rec["prob_"+ss]["repr"]),axis=1).flatten()
        return df

    # shape = (snap, nbit)
    def nsnapxnbit_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["bit"] =  np.tile(np.arange(sh[1]),sh[0])
        df["type"] = np.tile(np.concatenate((\
                        np.repeat("surv",rec["max_ls"]*rec["n_base"]),\
                        np.repeat("repr",(rec["max_ls"]-rec["maturity"])*rec["n_base"]),\
                        np.repeat("neut",rec["n_neutral"]*rec["n_base"]))),\
                        sh[0])
        for key in keys:
            df[key] = rec[key].flatten()
        return df

    # special: sliding window
    def sliding_window_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["bit"] =  np.tile(np.arange(sh[1]),sh[0])
        for key in keys:
            df[key] = rec[key].flatten()
        return df

    # shape = (nsnap, nloci)
    def nsnapxnloci_df(rec):
        df = pd.DataFrame()
        sh = rec["mean_gt"]["a"].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["locus"] =  np.tile(np.arange(sh[1]),sh[0])
        df["type"] = np.tile(np.concatenate((\
                        np.repeat("surv",rec["max_ls"]),\
                        np.repeat("repr",(rec["max_ls"]-rec["maturity"])),\
                        np.repeat("neut",rec["n_neutral"]))),\
                        sh[0])
        df["mean_gt"] = rec["mean_gt"]["a"].flatten()
        df["var_gt"] = rec["var_gt"]["a"].flatten()
        return df

    # shape = (nsnap,)
    def nsnapx1_df(rec, keys):
        df = pd.DataFrame()
        df["snap"] = np.arange(rec[keys[0]].size)
        for key in keys:
            df[key] = rec[key]
        return df

    # create dir in outpath and update outpath
    outpath = os.path.join(outpath,rec["output_prefix"]+"_csv_files")
    if not os.path.exists(outpath): os.makedirs(outpath)

    # output to csv
    single_df(rec, single_keys).to_csv(os.path.join(outpath,"single.csv"),index=False)
    nstagex1_df(rec).to_csv(os.path.join(outpath,"nstage-x-1.csv"),index=False)
    # output age dist only if it was recorded for all stages
    if rec["age_dist_N"]=="all":
        nstagexmaxls_df(rec,nstagexmaxls_keys).to_csv(os.path.join(outpath,\
            "nstage-x-maxls.csv"),index=False)
        maxlsx1_df(rec,maxlsx1_keys).to_csv(os.path.join(outpath,"maxls-x-1.csv"),index=False)
    nsnapxmaxls_df(rec,nsnapxmaxls_keys).to_csv(os.path.join(outpath,\
            "nsnap-x-maxls.csv"),index=False)
    nsnapxnbit_df(rec,nsnapxnbit_keys).to_csv(os.path.join(outpath,\
            "nsnap-x-nbit.csv"),index=False)
    sliding_window_df(rec,sliding_window_keys).to_csv(os.path.join(outpath,\
            "sliding_window.csv"),index=False)
    nsnapxnloci_df(rec).to_csv(os.path.join(outpath,"nsnap-x-nloci.csv"),index=False)
    nsnapx1_df(rec,nsnapx1_keys).to_csv(os.path.join(outpath,"nsnap-x-1.csv"),index=False)

    if verbose: print get_runtime(starttime, timenow(False), "Runtime")

############
### plot ###
############

def plot(record_file, verbose, outpath=""):
    a = Plotter(record_file, verbose)
    a.generate_figures()
    a.save_figures()
