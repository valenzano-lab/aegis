import pyximport; pyximport.install()
from .functions import chance, init_ages, init_genomes, init_generations
from .Population import Population, Outpop
from .Config import Config, Infodict
from .Record import Record
#import numpy as np
#import scipy.stats as st
