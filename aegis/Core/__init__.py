import os, shutil, numpy as np
from aegis.Core.functions import chance, quantile, fivenum, init_gentimes,\
        init_ages, init_genomes, init_generations, deep_key, deep_eq, make_windows,\
        correct_r_rate, make_mapping
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

def run(config_file, report_n, verbose):
    """Execute a complete simulation from a specified config file."""
    s = Simulation(config_file, report_n, verbose)
    s.execute()
    s.finalise()

def getconfig(outpath):
    """Create a default config file at the specified destination."""
    dirpath = os.path.dirname(os.path.realpath(__file__))
    inpath = os.path.join(dirpath, "config_default.py")
    shutil.copyfile(inpath, outpath)

def getrseed(inpath, outpath):
    """Get prng from a record object."""
    fin = open(inpath, "r")
    record = pickle.load(fin)
    fin.close()
    fout = open(outpath, "w")
    pickle.dump(record["random_seed"], fout)
    fout.close()

def getrecinfo(inpath, outpath):
    """Get information on record object."""
    rec_name = inpath.split('/')[-1] # get record name
    # load record
    infile = open(inpath)
    rec = pickle.load(infile)
    infile.close()
    # formatting
    n_tabs = 3
    ml_type = 4
    ml_name = 4
    ml_shape = 5
    ml_subkeys = 7
    for key in rec.keys():
        ml_type = max(ml_type, len(str(type(rec[key]))))
        ml_name = max(ml_name, len(key))
        if isinstance(rec[key], np.ndarray): ml_shape = max(ml_shape,\
                len(str(rec[key].shape)))
        elif isinstance(rec[key], list): ml_shape = max(ml_shape,\
                len(str(len(rec[key]))))
        elif isinstance(rec[key], tuple): ml_shape = max(ml_shape,\
                len(str(len(rec[key]))))
        elif isinstance(rec[key], dict): ml_subkeys = max(ml_subkeys,\
                len(str(rec[key].keys())))
    ll = []
    ll_title = ['type'+(ml_type-4)*' '+'|'+n_tabs*'\t'+\
            'name'+(ml_name-4)*' '+'|'+n_tabs*'\t'+\
            'shape']
    ll_sep = ['-'*ml_type+'|'+'-'*(ml_name+4*n_tabs)+'|'+'-'*(ml_shape-1+4*n_tabs)]
    ll_dicts = []
    ll_dicts_sep = ['-'*ml_type+'|'+'-'*(ml_name+4*n_tabs)+'|'+\
            '-'*(ml_name+4*n_tabs)]
    ll_dicts_title = ['type'+(ml_type-4)*' '+'|'+n_tabs*'\t'+\
            'name'+(ml_name-4)*' '+'|'+n_tabs*'\t'+\
            'subkeys']
    for key in rec.keys():
        s = str(type(rec[key]))
        s += ' '*(ml_type-len(s))
        s += '|'+(n_tabs * '\t')
        s += key
        s += ' '*(ml_name-len(key))
        s += '|'+(n_tabs * '\t')
        if isinstance(rec[key], dict):
            s += str(rec[key].keys())
            ll_dicts.append(s)
        else:
            if isinstance(rec[key],np.ndarray): ss = str(rec[key].shape)
            elif isinstance(rec[key],list): ss = str(len(rec[key]))
            elif isinstance(rec[key],tuple): ss = str(len(rec[key]))
            else: ss= '/'
            s += ss+' '*(ml_shape-len(ss))
            ll.append(s)
    # finalise
    ll.sort()
    ll = ll_title + ll_sep + ll
    ll_dicts.sort()
    ll_dicts = ll_dicts_title + ll_dicts_sep + ll_dicts
    out = '\n'.join([rec_name+' record entries\n']+ll+['\ndictionaries\n']+ll_dicts)
    # write to file
    outfile = open(outpath, 'w')
    outfile.write(out)
    outfile.close()

def plot(record_file):
    a = Plotter(record_file)
    a.generate_plots()
    a.save_plots()

def plot_n1_sliding_window(record_file, wsize):
    a = Plotter(record_file)
    a.compute_n1_windows(wsize)
    a.gen_save_single("n1_mean_sliding_window")
    a.gen_save_single("n1_var_sliding_window")
    a.gen_save_single("n1_mean_sliding_window_grid")
    a.gen_save_single("n1_var_sliding_window_grid")
