import pyximport; pyximport.install()
from aegis.Core.functions import chance, init_ages, init_genomes, init_generations
from aegis.Core.Population import Population, Outpop
from aegis.Core.Config import Config, Infodict, deepeq
from aegis.Core.Record import Record
from aegis.Core.Run import Run
from aegis.Core.Simulation import Simulation
