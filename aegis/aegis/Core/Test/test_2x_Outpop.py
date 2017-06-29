from aegis.Core import Config, Population, Outpop # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
import pytest, random, copy
import numpy as np

from test_1_Config import conf 
from test_2a_Population_init import pop
# (will run descendent tests for all parameterisations)

##############
## FIXTURES ##
##############

###########
## TESTS ##
###########

class TestOutpop:
    """Test Outpop object initialisation and methods."""

    def test_outpop_init(self, pop):
        o = Outpop(pop)
        assert o.params() == pop.params()
        #...
