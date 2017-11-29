import pyximport, os, shutil; pyximport.install()
from aegis.Core.functions import chance, init_ages, init_genomes, init_generations
from aegis.Core.Population import Population, Outpop
from aegis.Core.Config import Config, Infodict, deepeq
from aegis.Core.Record import Record
from aegis.Core.Run import Run
from aegis.Core.Simulation import Simulation
from aegis.Core.Plotter import Plotter

def run(config_file, report_n, verbose):
    """Execute a complete simulation from a specified config file."""
    s = Simulation(config_file, report_n, verbose)
    s.execute() # TODO: Implement parallelisation
    s.finalise()

def getconfig(outpath):
    """Create a default config file at the specified destination."""
    dirpath = os.path.dirname(os.path.realpath(__file__))
    inpath = os.path.join(dirpath, "config_default.py")
    shutil.copyfile(inpath, outpath) 

def plot(record_file):
    a = Plotter(record_file)
    a.generate_plots()
    a.save_plots()
