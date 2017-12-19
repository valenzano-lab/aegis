from aegis.Core import Config, Population, Outpop # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
import pytest, random, copy
import numpy as np

from test_1_Config import conf, conf_path, ran_str
from test_2a_Population_init import pop
# (will run descendent tests for all parameterisations)

##############
## FIXTURES ##
##############

@pytest.fixture(scope="module")
def opop(request, pop):
    """Convert a sample population into Outpop format."""
    return Outpop(pop)

###########
## TESTS ##
###########

class TestOutpop:
    """Test Outpop object initialisation and methods."""

    def test_outpop_init(self, pop):
        o = Outpop(pop)
        # Test if Outpop parameters and content match original population
        assert o.params() == pop.params()
        for a in ["genmap", "ages", "genomes", "generations"]:
            assert np.array_equal(getattr(o, a), getattr(pop,a))
        assert o.N == pop.N
        assert type(o) is not type(pop)
        # Test that new Outpop object is independent of parent population
        o.ages[0] = -1
        o.generations[0] = -1
        o.genomes[0,0] = -1
        assert np.array_equal(o.genmap, pop.genmap)
        for a in ["ages", "genomes", "generations"]:
            assert not np.array_equal(getattr(o, a), getattr(pop,a))

    def test_outpop_params(self, pop, opop):
        """Test that params returns (at least) the
        required information."""
        #if conf["setup"] == "random": return
        o,p = opop.params(), pop.params()
        K = o.keys()
        for s in ["repr_mode", "chr_len", "n_base", "max_ls", "maturity",
                "g_dist", "repr_offset", "neut_offset"]:
            assert s in K
        for k in K:
            a,b,c = getattr(opop, k), o[k], p[k]
            assert np.array_equal(a,b) if type(a) is np.ndarray else a == b
            assert np.array_equal(b,c) if type(b) is np.ndarray else b == c

    def test_outpop_toPop(self, pop, opop):
        """Test that toPop correctly reverts an object to its
        cythonised state."""
        # Test that opop.toPop() is equivalent to pop
        p1, p2 = pop, opop.toPop()
        assert p1.params() == p2.params()
        for a in ["genmap", "ages", "genomes", "generations"]:
            assert np.array_equal(getattr(p1, a), getattr(p2,a))
        assert p1.N == p2.N
        assert type(p1) is type(p2)
        # Test that new population is independent of parent Outpop
        p2.ages[0] = -1
        p2.generations[0] = -1
        p2.genomes[0,0] = -1
        assert np.array_equal(p2.genmap, opop.genmap)
        for a in ["ages", "genomes", "generations"]:
            assert not np.array_equal(getattr(opop, a), getattr(p2,a))

    def test_outpop_clone(self, pop, opop):
        """Test if cloned Outpop is identical to parent Outpop,
        by comparing params, ages, genomes."""
        #if conf["setup"] == "random": return
        opop2 = opop.clone()
        opop2.generations[0] = 1
        opop3 = opop2.clone()
        assert opop3.params() == opop2.params()
        for a in ["genmap", "ages", "genomes", "generations"]:
            assert np.array_equal(getattr(opop3, a), getattr(opop2,a))
        assert opop3.N == opop2.N
        # Test that populations are now independent
        opop3.ages[0] = -1
        opop3.generations[0] = -1
        opop3.genomes[0,0] = -1
        assert np.array_equal(opop2.genmap, opop3.genmap)
        for a in ["ages", "genomes", "generations"]:
            assert not np.array_equal(getattr(opop3, a), getattr(opop2,a))
