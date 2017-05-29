from aegis.Core import Config, Population, Outpop # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
import pytest, random, copy
import numpy as np

from test_1_Config import conf 
# (will run descendent tests for all parameterisations)

##############
## FIXTURES ##
##############

@pytest.fixture()
def pop(request, conf):
    """Create a sample population from the default configuration."""
    return Population(conf["params"], conf["genmap"], init_ages(),
            init_genomes(), init_generations())

class TestPopulation:
    """Test population object methods."""

    # Initialisation
    def test_init_population_blank(self, conf):
        """Test that population parameters are correct for random and
        nonrandom ages, assuming that genome generation is correct."""
        rtol = 0.02
        x = random.random()
        c = copy.deepcopy(conf)
        c["params"]["g_dist"] = {"s":x,"r":x,"n":x}
        #c["params"]["start_pop"] = 2000
        #c["params"]["g_dist"] = {"s":x,"r":x,"n":x}
        pop = Population(c["params"], c["genmap"], init_ages(),
                init_genomes(), init_generations())
        #print np.mean(pop_b.ages)
        #print abs(np.mean(pop_b.ages)-pop_b.maxls/2)
        # Check basic parameters
        assert pop.repr_mode == c["params"]["repr_mode"]
        assert pop.chr_len == c["params"]["chr_len"]
        assert pop.n_base == c["params"]["n_base"]
        assert pop.max_ls == c["params"]["max_ls"]
        assert pop.maturity == c["params"]["maturity"]
        assert pop.g_dist == c["params"]["g_dist"]
        assert pop.repr_offset == c["params"]["repr_offset"]
        assert pop.neut_offset == c["params"]["neut_offset"]
        assert np.array_equal(pop.genmap, c["genmap"])
        assert np.array_equal(pop.genmap_argsort, c["genmap_argsort"])
        # Check reproductive state
        recombine = pop.repr_mode in ['sexual','recombine_only']
        assort = pop.repr_mode in ['sexual','assort_only']
        assert pop.recombine == recombine
        assert pop.assort == assort
        # Check population state
        assert pop.N == c["params"]["start_pop"]
        assert np.isclose(np.mean(pop.genomes), x, atol=0.02)
        assert np.isclose(np.mean(pop.ages), np.mean(np.arange(pop.max_ls)),
                atol = 2.5)
        assert np.all(pop.generations == 0)

    def test_init_population_nonblank(self, conf):
        """Test that population is generated correctly when ages and/or
        genomes are provided."""
        # Set up test values
        if conf["setup"] == "random": return
        c = copy.deepcopy(conf)
        x = random.random()
        ages_r = np.random.randint(0, 10, c["params"]["start_pop"])
        genomes_r = chance(1-x**2, 
                (c["params"]["start_pop"],c["params"]["chr_len"])).astype(int)
        generations_r = np.random.randint(0, 10, c["params"]["start_pop"])
        x = random.random()
        c["params"]["g_dist"] = {"s":x,"r":x,"n":x}
        # Initialise populations
        def pop(ages, genomes, generations):
            return Population(c["params"], c["genmap"], ages, genomes, generations)
        pops = [pop(ages_r, genomes_r, generations_r),
                pop(ages_r, genomes_r, init_generations()),
                pop(ages_r, init_genomes(), generations_r),
                pop(ages_r, init_genomes(), init_generations()),
                pop(init_ages(), genomes_r, generations_r),
                pop(init_ages(), genomes_r, init_generations()),
                pop(init_ages(), init_genomes(), generations_r),
                pop(init_ages(), init_genomes(), init_generations())]
        # Check params as for default state
        for pop in pops:
            assert pop.repr_mode == c["params"]["repr_mode"]
            assert pop.chr_len == c["params"]["chr_len"]
            assert pop.n_base == c["params"]["n_base"]
            assert pop.max_ls == c["params"]["max_ls"]
            assert pop.maturity == c["params"]["maturity"]
            assert pop.g_dist == c["params"]["g_dist"]
            assert pop.repr_offset == c["params"]["repr_offset"]
            assert pop.neut_offset == c["params"]["neut_offset"]
            assert np.array_equal(pop.genmap, c["genmap"])
            assert np.array_equal(pop.genmap_argsort, c["genmap_argsort"])
            recombine = pop.repr_mode in ['sexual','recombine_only']
            assort = pop.repr_mode in ['sexual','assort_only']
            assert pop.recombine == recombine
            assert pop.assort == assort
            assert pop.N == c["params"]["start_pop"]
        # Check ages
        for n in [0,1,2,3]: 
            assert np.array_equal(ages_r, pops[n].ages)
        for n in [4,5,6,7]:
            assert not np.array_equal(ages_r, pops[n].ages)
            assert np.isclose(np.mean(pops[n].ages),
                    np.mean(np.arange(pops[n].max_ls)), atol=2.5)
        # Check genomes
        for n in [0,1,4,5]:
            assert np.array_equal(genomes_r, pops[n].genomes)
        for n in [2,3,6,7]:
            assert not np.array_equal(genomes_r, pops[n].genomes)
            assert np.isclose(np.mean(pops[n].genomes), x, atol=0.02)
        # Check generations
        for n in [0,2,4,6]:
            assert np.array_equal(generations_r, pops[n].generations)
        for n in [1,3,5,7]:
            assert not np.array_equal(generations_r, pops[n].generations)
            assert np.all(pops[n].generations == 0)

    def test_popgen_independence(self, pop, conf):
        #if conf["setup"] == "random": return
        """Test that generating a population from another and then manipulating
        the cloned population does not affect original (important for
        reproduction)."""
        P1 = pop.clone()
        P2 = Population(P1.params(), P1.genmap, P1.ages, P1.genomes,
                P1.generations)
        P3 = Population(P1.params(), P1.genmap, P1.ages[:100], 
                P1.genomes[:100], P1.generations[:100])
        P4 = Population(P3.params(), P3.genmap, P3.ages, P3.genomes,
                P3.generations)
        P2.mutate(0.5, 1)
        P4.mutate(0.5, 1)
        # Test ages
        assert np.array_equal(pop.ages, P1.ages)
        assert np.array_equal(pop.ages, P2.ages)
        assert np.array_equal(pop.ages[:100], P3.ages)
        assert np.array_equal(pop.ages[:100], P4.ages)
        # Test genomes
        assert np.array_equal(pop.genomes, P1.genomes)
        assert np.array_equal(pop.genomes[:100], P3.genomes)
        assert pop.genomes.shape == P2.genomes.shape
        assert pop.genomes[:100].shape == P4.genomes.shape
        assert not np.array_equal(pop.genomes, P2.genomes)
        assert not np.array_equal(pop.genomes[:100], P4.genomes)

    def test_params(self, pop, conf):
        """Test that params returns (at least) the
        required information."""
        p = pop.params()
        for k in p.keys():
            a,b = getattr(pop, k), p[k]
            assert np.array_equal(a,b) if type(a) is np.ndarray else a == b
