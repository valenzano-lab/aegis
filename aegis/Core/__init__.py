import os, shutil, numpy as np
from aegis.Core.functions import chance, quantile, fivenum, init_gentimes
from aegis.Core.functions import init_ages, init_genomes, init_generations
from aegis.Core.functions import deep_key, deep_eq
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
