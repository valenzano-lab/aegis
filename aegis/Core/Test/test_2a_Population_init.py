from aegis.Core import Config, Population # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
from aegis.Core import init_gentimes
import pytest, random, copy
import numpy as np

##############
## FIXTURES ##
##############

from test_1_Config import conf_naive, conf, conf_path, ran_str, gen_trseed
# will run descendent tests for all parameterisations

@pytest.fixture(scope="module")
def pop(request, conf):
    """Create a sample population from the default configuration."""
    return Population(conf["params"], conf["genmap"], conf["mapping"], init_ages(),
            init_genomes(), init_generations(), init_gentimes())

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
        # Permute parameter values (other than g_dist,repr_mode and prng)
        for a in pop.params().keys():
            if a in ["repr_mode","g_dist", "prng"]: continue
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
            if a=="prng":
                assert getattr(pop, a) == getattr(pop2, a) \
                        == getattr(pop3, a)
            elif type(getattr(pop, a)) is np.ndarray:
                assert np.array_equal(getattr(pop2, a), getattr(pop3, a))
                assert not np.array_equal(getattr(pop, a), getattr(pop3, a))
            else:
                assert getattr(pop2, a) == getattr(pop3, a)
                assert getattr(pop, a) != getattr(pop3, a)

    def test_set_attributes_recombine_assort(self, pop):
        """Test correct assignment of recombine and assort attributes
        from reproductive mode."""
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

    def test_initial_size_compatible(self, pop, conf):
        """Test that initial population size is computed correctly
        under different valid conditions."""
        # New pop with altered members and size
        pop2 = pop.clone()
        drop = random.sample(xrange(pop2.N), random.randrange(20))
        pop2.subtract_members(drop)
        pop2.N = 0
        # State 1: all new
        p = copy.deepcopy(conf["params"])
        p["start_pop"] += random.randint(1,20)
        pop2.set_initial_size(p, init_ages(),
                init_genomes(), init_generations(), init_gentimes())
        assert pop2.N == p["start_pop"]
        # State 2: none new
        pop2.N = 0
        pop2.set_initial_size(pop.params(), pop.ages, pop.genomes,
                pop.generations, pop.gentimes)
        assert pop2.N == pop.N
        # State 3: two new
        pop3 = pop.clone()
        n3 = pop3.N
        drop = random.sample(xrange(pop3.N), random.randrange(10))
        pop3.subtract_members(drop)
        pop2.N = 0
        pop2.set_initial_size(pop3.params(), pop3.ages,
                init_genomes(), init_generations(), init_gentimes())
        assert pop2.N == n3 - len(drop)
        n3 = pop3.N
        drop = random.sample(xrange(pop3.N), random.randrange(10))
        pop3.subtract_members(drop)
        pop2.N = 0
        pop2.set_initial_size(pop3.params(), init_ages(),
                pop3.genomes, init_generations(), init_gentimes())
        assert pop2.N == n3 - len(drop)
        n3 = pop3.N
        drop = random.sample(xrange(pop3.N), random.randrange(10))
        pop3.subtract_members(drop)
        pop2.N = 0
        pop2.set_initial_size(pop3.params(), init_ages(),
                init_genomes(), pop3.generations, init_gentimes())
        assert pop2.N == n3 - len(drop)
        # State 4: one new
        pop4 = pop.clone()
        n4 = pop4.N
        drop = random.sample(xrange(pop4.N), random.randrange(10))
        pop4.subtract_members(drop)
        pop2.N = 0
        pop2.set_initial_size(pop4.params(), pop4.ages,
                pop4.genomes, init_generations(), init_gentimes())
        assert pop2.N == n4 - len(drop)
        n4 = pop4.N
        drop = random.sample(xrange(pop4.N), random.randrange(10))
        pop4.subtract_members(drop)
        pop2.N = 0
        pop2.set_initial_size(pop4.params(), pop4.ages,
                init_genomes(), pop4.generations, init_gentimes())
        assert pop2.N == n4 - len(drop)
        n4 = pop4.N
        drop = random.sample(xrange(pop4.N), random.randrange(10))
        pop4.subtract_members(drop)
        pop2.N = 0
        pop2.set_initial_size(pop4.params(), init_ages(),
                pop4.genomes, pop4.generations, init_gentimes())
        assert pop2.N == n4 - len(drop)

    def test_initial_size_compatible(self, pop):
        """Test that set_initial_size returns an appropriate error
        when incompatible data is given."""
        # New pop with altered members and size
        pop2 = pop.clone()
        drop = random.sample(xrange(pop2.N), random.randrange(20))
        pop2.subtract_members(drop)
        pop2.N = 0
        pop3 = pop.clone()
        def rsp3(n): return random.sample(xrange(pop3.N), n)
        pop3.ages = np.delete(pop3.ages, rsp3(1), 0)
        pop3.genomes = np.delete(pop3.genomes, rsp3(2), 0)
        pop3.generations = np.delete(pop3.generations, rsp3(3), 0)
        with pytest.raises(ValueError):
            pop2.set_initial_size(pop3.params(), pop3.ages, pop3.genomes,
                    init_generations(), init_gentimes())
        with pytest.raises(ValueError):
            pop2.set_initial_size(pop3.params(), pop3.ages, init_genomes(),
                    pop3.generations, init_gentimes())
        with pytest.raises(ValueError):
            pop2.set_initial_size(pop3.params(), init_ages(), pop3.genomes,
                    pop3.generations, init_gentimes())
        with pytest.raises(ValueError):
            pop2.set_initial_size(pop3.params(), pop3.ages, pop3.genomes,
                    pop3.generations, init_gentimes())

    def test_fill(self, conf):
        """Test that Population.fill correctly reads in member data
        when provided and generates them otherwise."""
        # Set up random test inputs
        c = copy.deepcopy(conf)
        c["start_pop"] *= 10
        x = random.random()
        ages_r = np.random.randint(0, 10, c["params"]["start_pop"])
        genomes_r = chance(1-x**2,
                (c["params"]["start_pop"],c["params"]["chr_len"]*2)).astype(int)
        generations_r = np.random.randint(0, 10, c["params"]["start_pop"])
        x = random.random()
        c["params"]["g_dist"] = {"s":x,"r":x,"n":x}
        # Initialise populations
        def pop(ages, genomes, generations, gentimes, targets):
            return Population(c["params"], c["genmap"], c["mapping"], ages, genomes,
                    generations, gentimes, targets)
        tgts = np.zeros(c["params"]["start_pop"]).astype(int)
        tgts[:200] = 1
        tgts = tgts.astype(bool)
        pops = [pop(ages_r, genomes_r, generations_r, init_gentimes(),tgts),
                pop(ages_r, genomes_r, init_generations(), init_gentimes(),tgts),
                pop(ages_r, init_genomes(), generations_r, init_gentimes(),tgts),
                pop(ages_r, init_genomes(), init_generations(), init_gentimes(),tgts),
                pop(init_ages(), genomes_r, generations_r, init_gentimes(),tgts),
                pop(init_ages(), genomes_r, init_generations(), init_gentimes(),tgts),
                pop(init_ages(), init_genomes(), generations_r, init_gentimes(),tgts),
                pop(init_ages(), init_genomes(), init_generations(), init_gentimes(),tgts)]
        tgts = np.arange(tgts.sum())
        # Check ages
        for n in [0,1,2,3]:
            assert np.array_equal(ages_r[tgts], pops[n].ages)
        for n in [4,5,6,7]:
            assert not np.array_equal(ages_r[tgts], pops[n].ages)
            m = np.mean(pops[n].ages)/np.mean(np.arange(pops[n].max_ls))
            assert np.isclose(m, 1, atol=0.1)
        # Check genomes
        for n in [0,1,4,5]:
            assert (pops[n].N, pops[n].chr_len*2) == pops[n].genomes.shape
            assert np.array_equal(genomes_r[tgts], pops[n].genomes)
        for n in [2,3,6,7]:
            assert (pops[n].N, pops[n].chr_len*2) == pops[n].genomes.shape
            assert not np.array_equal(genomes_r[tgts], pops[n].genomes)
            assert np.isclose(np.mean(pops[n].genomes), x, atol=0.02)
        # Check generations
        for n in [0,2,4,6]:
            assert np.array_equal(generations_r[tgts], pops[n].generations)
        for n in [1,3,5,7]:
            assert not np.array_equal(generations_r[tgts], pops[n].generations)
            assert np.all(pops[n].generations == 0)
        # Check loci
        for p in pops:
            assert np.array_equal(p.loci[tgts], p.sorted_loci())
        # Check gentimes
        for n in xrange(8):
            assert np.all(pops[n].gentimes == 0L)

    def test_population_init(self, pop):
        """Test that Population.__init__ applies the correct methods..."""
        pop2 = pop.clone()
        # Delete everything
        del pop2.genmap, pop2.genmap_argsort
        pop2.repr_mode, pop2.recombine, pop2.assort = ["",-1,-1]
        pop2.chr_len, pop2.maturity, pop2.max_ls, pop2.n_base, pop2.n_neutral = [0,0,0,0,0]
        pop2.g_dist = {}
        pop2.repr_offset, pop2.neut_offset, pop2.N = [0,0,0]
        del pop2.ages, pop2.genomes, pop2.generations, pop2.gentimes
        # Refill
        pop2.set_genmap(pop.genmap)
        pop2.set_attributes(pop.params())
        pop2.set_initial_size(pop.params(), pop.ages,
                pop.genomes, pop.generations, pop.gentimes)
        pop2.fill(pop.ages, pop.genomes, pop.generations, pop.gentimes)
        # Test that pop2 and pop are identical
        for a in dir(pop):
            print(a)
            if callable(getattr(pop, a)): continue # Skip functions
            if type(getattr(pop, a)) is np.ndarray:
                assert np.array_equal(getattr(pop, a), getattr(pop2, a))
            else:
                assert getattr(pop, a) == getattr(pop2, a)

    def test_popgen_independence(self, pop, conf):
        """Test that generating a population from another and then manipulating
        the cloned population does not affect original (important for
        reproduction)."""
        P1 = pop.clone()
        P2 = Population(P1.params(), P1.genmap, P1.mapping, P1.ages, P1.genomes,
                P1.generations, P1.gentimes)
        P3 = Population(P1.params(), P1.genmap, P1.mapping, P1.ages[:100],
                P1.genomes[:100], P1.generations[:100], P1.gentimes[:100])
        P4 = Population(P3.params(), P3.genmap, P3.mapping, P3.ages, P3.genomes,
                P3.generations, P3.gentimes)
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
        # Test generations
        assert np.array_equal(pop.generations, P1.generations)
        assert np.array_equal(pop.generations, P2.generations)
        assert np.array_equal(pop.generations[:100], P3.generations)
        assert np.array_equal(pop.generations[:100], P4.generations)
        # Test gentimes
        assert np.array_equal(pop.gentimes, P1.gentimes)
        assert np.array_equal(pop.gentimes, P2.gentimes)
        assert np.array_equal(pop.gentimes[:100], P3.gentimes)
        assert np.array_equal(pop.gentimes[:100], P4.gentimes)

    # TODO: Tests for individual parts of initialisation

    def test_make_genome_array(self, pop):
        """Test that genome array is of the correct size and that
        the loci are distributed correctly."""
        precision = 0.05
        # Define testing function
        def test_mga(genmap):
            # Set up population
            pop2 = pop.clone()
            pop2.N = 1000
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
            np.arange(24) + pop.repr_offset,
            np.arange(5) + pop.neut_offset), 0)
        genmap2 = np.copy(genmap1)
        random.shuffle(genmap2)
        test_mga(genmap1)
        test_mga(genmap2)
