from aegis.Core import Config, Population, Outpop # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
import pytest, random, copy
import numpy as np

# Import fixtures
from test_1_Config import conf 
from test_2a_Population_init import pop

class TestPopulationReComb:
    """Test methods of Population object relating to rearrangement and
    combination of populations."""

    def test_params(self, pop, conf):
        """Test that params returns (at least) the
        required information."""
        #if conf["setup"] == "random": return
        p = pop.params()
        K = p.keys()
        for s in ["repr_mode", "chr_len", "n_base", "max_ls", "maturity",
                "g_dist", "repr_offset", "neut_offset"]:
            assert s in K
        for k in K:
            a,b = getattr(pop, k), p[k]
            assert np.array_equal(a,b) if type(a) is np.ndarray else a == b

    def test_clone(self, pop, conf):
        """Test if cloned population is identical to parent population,
        by comparing params, ages, genomes."""
        #if conf["setup"] == "random": return
        pop2 = pop.clone()
        pop2.generations[0] = 1
        pop3 = pop2.clone()
        assert pop3.params() == pop2.params()
        for a in ["genmap", "ages", "genomes", "generations"]:
            assert np.array_equal(getattr(pop3, a), getattr(pop2,a))
        assert pop3.N == pop2.N
        # Test that populations are now independent
        pop3.ages[0] = -1
        pop3.generations[0] = -1
        pop3.genomes[0,0] = -1
        assert np.array_equal(pop2.genmap, pop3.genmap)
        for a in ["ages", "genomes", "generations"]:
            assert not np.array_equal(getattr(pop3, a), getattr(pop2,a))

    def test_attrib_rep(self, pop):
        """Test repetition of function over attributes."""
        pop2 = pop.clone()
        def f1(x): return x+1
        def f2(x): return x-1
        pop2.attrib_rep(f1)
        for a in ["ages", "genomes", "generations"]:
            assert np.array_equal(getattr(pop, a)+1, getattr(pop2,a))
        assert pop.params() == pop2.params()
        assert pop.N == pop2.N
        pop2.attrib_rep(f2)
        for a in ["ages", "genomes", "generations"]:
            assert np.array_equal(getattr(pop, a), getattr(pop2,a))
        assert pop.params() == pop2.params()
        assert pop.N == pop2.N

    def test_shuffle(self, pop, conf):
        """Test if all ages, therefore individuals, present before the
        shuffle are also present after it."""
        #if conf["setup"] == "random": return
        pop2 = pop.clone()
        pop2.generations[0] = 1
        pop3 = pop2.clone()
        pop3.shuffle()
        assert not np.array_equal(pop3.ages, pop2.ages)
        assert not np.array_equal(pop3.generations, pop2.generations)
        assert not np.array_equal(pop3.genomes, pop2.genomes)
        pop3.ages.sort()
        pop2.ages.sort()
        pop3.generations.sort()
        pop2.generations.sort()
        assert not np.array_equal(pop3.genomes, pop2.genomes)
        assert np.array_equal(pop3.ages, pop2.ages)
        assert np.array_equal(pop3.generations, pop2.generations)

    def test_subset_members(self, pop):
        """Test if a population subset is correctly produced from a
        boolean retention vector."""
        # Case 1: all True
        pop2 = pop.clone()
        v = np.ones(pop2.N, dtype=bool)
        pop2.subset_members(v)
        for a in ["ages", "genomes", "generations"]:
            assert np.array_equal(getattr(pop, a), getattr(pop2,a))
        assert pop.params() == pop2.params()
        assert pop.N == pop2.N
        # Case 2: all False
        pop3 = pop.clone()
        v = np.zeros(pop3.N, dtype=bool)
        pop3.subset_members(v)
        for a in ["ages", "genomes", "generations"]:
            assert len(getattr(pop3,a)) == 0
        assert pop3.N == 0
        # Case 3: alternating
        pop4 = pop.clone()
        v = [True, False] * (pop4.N/2)
        if pop.N % 2 == 1: v += [True]
        v = np.array(v)
        pop4.subset_members(v)
        for a in ["ages", "genomes", "generations"]:
            assert np.array_equal(getattr(pop, a)[0::2],
                    getattr(pop4,a))
        assert pop.params() == pop4.params()
        if pop.N % 2 == 1:
            n4 = (pop4.N - 1) * 2 + 1
        else:
            n4 = pop4.N * 2
        assert pop.N == n4
        # Case 4: random
        pop5 = pop.clone()
        v = np.random.randint(0,2,pop5.N,dtype=bool)
        pop5.subset_members(v)
        for a in ["ages", "genomes", "generations"]:
            assert np.array_equal(getattr(pop, a)[v],
                    getattr(pop5,a))
        assert pop.params() == pop5.params()
        assert pop5.N == np.sum(v)

    def test_subtract_members(self, pop):
        """Test if a population is correctly reduced through removal
        of specific members."""
        # Identify sample
        n = random.randint(1, pop.N-1) # Size of sample
        i = random.sample(xrange(pop.N), n) # Individuals to be removed
        # Apply to new population
        pop2 = pop.clone()
        print n
        print pop2.N
        pop2.subtract_members(i)
        print pop2.N
        # Check result
        for a in ["ages", "genomes", "generations"]:
            print a
            assert np.array_equal(np.delete(getattr(pop, a), i, 0),
                    getattr(pop2,a))

    def test_add_members(self, pop):
        """Test if a population is successfully appended to the receiver
        population, which remains unchanged, by appending a population to
        itself."""
        pop_a = pop.clone()
        pop_b = pop.clone()
        pop_b.add_members(pop_a)
        assert np.array_equal(pop_b.ages, np.tile(pop_a.ages,2))
        assert np.array_equal(pop_b.genomes, np.tile(pop_a.genomes,(2,1)))
        assert pop_b.N == 2*pop_a.N

    def test_subset_clone(self, pop):
        # Initialise
        pop2 = pop.clone()
        v = np.random.randint(0,2,pop2.N,dtype=bool)
        pop3 = pop2.subset_clone(v)
        # Test pop3 != pop2
        assert pop3.params() == pop2.params()
        assert pop3.N == np.sum(v)
        assert pop2.N == pop.N
        for a in ["ages", "genomes", "generations"]:
            assert not np.array_equal(getattr(pop3, a), getattr(pop2,a))
        # Test that subsetting pop2 by same index produces same result
        pop2.subset_members(v)
        assert pop3.params() == pop2.params()
        assert pop3.N == pop2.N
        for a in ["ages", "genomes", "generations"]:
            assert np.array_equal(getattr(pop3, a), getattr(pop2,a))
        # Test that populations are now independent
        pop3.ages[0] = -1
        pop3.generations[0] = -1
        pop3.genomes[0,0] = -1
        for a in ["ages", "genomes", "generations"]:
            assert not np.array_equal(getattr(pop3, a), getattr(pop2,a))

class TestPopulationIncrement:
    """Test methods of Population object relating to incrementation
    of values describing individuals in the population."""

    def test_increment_ages(self, pop, conf):
        """Test if all ages are incremented by one."""
        #if conf["setup"] == "random": return
        P2 = pop.clone()
        P2.increment_ages()
        assert np.array_equal(pop.ages+1, P2.ages)
 
    def test_increment_generations(self, pop, conf):
        """Test if all generations are incremented by one."""
        #if conf["setup"] == "random": return
        P2 = pop.clone()
        P2.increment_generations()
        assert np.array_equal(pop.generations+1, P2.generations)
