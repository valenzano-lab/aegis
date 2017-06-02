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

class TestPopulationInit:
    """Test Population object methods relating to initialisation."""

    def test_set_genmap(self, pop):
        """Test setting a population's genome map from a genmap array."""
        # Test basic setting
        pop2 = pop.clone()
        gm2 = np.copy(pop.genmap) + 1
        random.shuffle(gm2)
        pop2.set_genmap(gm2)
        assert np.array_equal(pop2.genmap, gm2)
        assert np.array_equal(pop2.genmap_argsort, np.argsort(gm2))
        assert not np.array_equal(pop2.genmap, pop.genmap)
        assert not np.array_equal(pop2.genmap_argsort, pop.genmap_argsort)
        # Test independence
        gm2[0] = -1
        assert not np.array_equal(pop2.genmap, gm2)

    def test_set_attributes_general(self, pop, conf):
        """Test inheritance of Population attributes from a params
        dictionary, and independence of these values after cloning."""
        pop2 = pop.clone()
        # Permute parameter values (other than g_dist or repr_mode)
        for a in pop.params().keys():
            if a in ["repr_mode","g_dist"]: continue
            setattr(pop2, a, getattr(pop2, a) + random.randint(1,11))
        # Permute g_dist
        for k in pop.g_dist.keys():
            pop2.g_dist[k] = pop2.g_dist[k] * random.random()/2
            assert pop2.g_dist[k] != pop.g_dist[k]
        # Permute repr_mode
        all_modes = ["sexual", "asexual", "assort_only", "recombine_only"]
        poss_modes = list(set(all_modes) - set([pop.repr_mode]))
        pop2.repr_mode = random.choice(poss_modes)
        # Check inheritance
        pop3 = pop.clone()
        pop3.set_attributes(pop2.params())
        for a in pop.params().keys():
            if type(getattr(pop, a)) is np.ndarray:
                assert np.array_equal(getattr(pop2, a), getattr(pop3, a))
                assert not np.array_equal(getattr(pop, a), getattr(pop3, a))
            else:
                assert getattr(pop2, a) == getattr(pop3, a)
                assert getattr(pop, a) != getattr(pop3, a)

    def test_set_attributes_recombine_assort(self, pop):
        pop2 = pop.clone()
        pop3 = pop.clone()
        # Check setting of .recombine and .assort
        pop3.recombine, pop3.assort = False, False
        pop2.repr_mode = "sexual"
        pop3.set_attributes(pop2.params())
        assert [pop3.recombine, pop3.assort] == [True, True]
        pop2.repr_mode = "recombine_only"
        pop3.set_attributes(pop2.params())
        assert [pop3.recombine, pop3.assort] == [True, False]
        pop2.repr_mode = "assort_only"
        pop3.set_attributes(pop2.params())
        assert [pop3.recombine, pop3.assort] == [False, True]
        pop2.repr_mode = "asexual"
        pop3.set_attributes(pop2.params())
        assert [pop3.recombine, pop3.assort] == [False, False]

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
    
    #! TODO: Tests for individual parts of initialisation

    def test_make_genome_array(self, pop):
        """Test that genome array is of the correct size and that
        the loci are distributed correctly."""
        precision = 0.05
        # Set up population
        pop2 = pop.clone()
        pop2.N = 1000
        # Define testing function
        def test_mga(genmap):
            loci = {
                "s":np.nonzero(genmap<pop2.repr_offset)[0],
                "r":np.nonzero(np.logical_and(genmap>=pop2.repr_offset,
                    genmap<pop2.neut_offset))[0],
                "n":np.nonzero(genmap>=pop2.neut_offset)[0]
                }
            pop2.genmap = genmap
            gd = dict([(x,random.random()) for x in ["s","r","n"]])
            pop2.g_dist = gd
            ga = pop2.make_genome_array()
            assert ga.shape == (pop2.N, 2*pop2.chr_len)
            b = pop2.n_base
            for k in loci.keys():
                pos = np.array([range(b) + x for x in loci[k]*b])
                pos = np.append(pos, pos + pop2.chr_len)
                tstat = abs(np.mean(ga[:,pos])-gd[k])
                assert tstat < precision
        # First with simple linear genmap, then with shuffled form
        genmap1 = np.concatenate((np.arange(25), 
            np.arange(24) + pop2.repr_offset,
            np.arange(5) + pop2.neut_offset), 0)
        genmap2 = np.copy(genmap1)
        random.shuffle(genmap2)
        test_mga(genmap1)
        test_mga(genmap2)
