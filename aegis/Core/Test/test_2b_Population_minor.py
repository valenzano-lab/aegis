from aegis.Core import Config, Population # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
import pytest, random, copy
import numpy as np

# Import fixtures
from test_1_Config import conf, conf_path, ran_str, gen_trseed
from test_2a_Population_init import pop

attrs_no_loci = ("ages", "genomes", "generations", "gentimes")
attrs = attrs_no_loci + ("loci",)
attrs_genmap = attrs + ("genmap",)

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
        for a in attrs_genmap:
            assert np.array_equal(getattr(pop3, a), getattr(pop2,a))
        assert pop3.N == pop2.N
        # Test that populations are now independent
        for a in attrs:
            getattr(pop3, a)[0] = -1
            assert not np.array_equal(getattr(pop3, a), getattr(pop2,a))
        assert np.array_equal(pop2.genmap, pop3.genmap)

    def test_clone2(self,pop):
        """Test that cloned populations are equal via the equal method."""
        pop2 = pop.clone()
        assert pop == pop2

    def test_attrib_rep(self, pop):
        """Test repetition of function over attributes."""
        pop2 = pop.clone()
        def f1(x): return x+1
        def f2(x): return x-1
        pop2.attrib_rep(f1)
        for a in attrs:
            assert np.array_equal(getattr(pop, a)+1, getattr(pop2,a))
        assert pop.params() == pop2.params()
        assert pop.N == pop2.N
        pop2.attrib_rep(f2)
        for a in attrs:
            assert np.array_equal(getattr(pop, a), getattr(pop2,a))
        assert pop.params() == pop2.params()
        assert pop.N == pop2.N

    # this can fail by chance when shuffle is identity
    def test_shuffle(self, pop, conf):
        """Test if all ages, therefore individuals, present before the
        shuffle are also present after it."""
        #if conf["setup"] == "random": return
        pop2 = pop.clone()
        pop2.generations[0] = 1
        pop2.gentimes[0] = 1
        pop3 = pop2.clone()
        pop3.shuffle()
        assert np.array_equal(pop2.genmap, pop3.genmap)
        for a in attrs:
            print a
            assert not np.array_equal(getattr(pop3, a), getattr(pop2, a))
        pop3.ages.sort()
        pop2.ages.sort()
        pop3.generations.sort()
        pop2.generations.sort()
        for a in attrs:
            if a in ("ages", "generations"):
                assert np.array_equal(getattr(pop3, a), getattr(pop2, a))
            else:
                assert not np.array_equal(getattr(pop3, a), getattr(pop2, a))

    def test_subset_members(self, pop):
        """Test if a population subset is correctly produced from a
        boolean retention vector."""
        # Case 1: all True
        pop2 = pop.clone()
        v = np.ones(pop2.N, dtype=bool)
        pop2.subset_members(v)
        for a in attrs:
            assert np.array_equal(getattr(pop, a), getattr(pop2,a))
        assert pop.params() == pop2.params()
        assert pop.N == pop2.N
        # Case 2: all False
        pop3 = pop.clone()
        v = np.zeros(pop3.N, dtype=bool)
        pop3.subset_members(v)
        for a in attrs:
            assert len(getattr(pop3,a)) == 0
        assert pop3.N == 0
        # Case 3: alternating
        pop4 = pop.clone()
        v = [True, False] * (pop4.N/2)
        if pop.N % 2 == 1: v += [True]
        v = np.array(v)
        pop4.subset_members(v)
        for a in attrs:
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
        for a in attrs:
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
        for a in attrs:
            print a
            assert np.array_equal(np.delete(getattr(pop, a), i, 0),
                    getattr(pop2,a))

    def test_add_members(self, pop):
        """Test if a population is successfully appended to the receiver
        population, which remains unchanged, by appending a population to
        itself."""
        pop2 = pop.clone()
        pop3 = pop.clone()
        pop3.add_members(pop2)
        assert pop3.N == 2*pop2.N
        for a in attrs:
            print a
            o2, o3 = getattr(pop2, a), getattr(pop3, a)
            print o2.shape, o3.shape
            shape = (2,) + (1,) * (np.ndim(o2)-1)
            assert np.array_equal(o3, np.tile(o2, shape))

    def test_subset_clone(self, pop):
        # Initialise
        pop2 = pop.clone()
        v = np.random.randint(0,2,pop2.N,dtype=bool)
        pop3 = pop2.subset_clone(v)
        # Test pop3 != pop2
        assert pop3.params() == pop2.params()
        assert pop3.N == np.sum(v)
        assert pop2.N == pop.N
        for a in attrs:
            assert not np.array_equal(getattr(pop3, a), getattr(pop2,a))
        # Test that subsetting pop2 by same index produces same result
        pop2.subset_members(v)
        assert pop3.params() == pop2.params()
        assert pop3.N == pop2.N
        for a in attrs:
            assert np.array_equal(getattr(pop3, a), getattr(pop2,a))
        # Test that populations are now independent
        for a in attrs:
            getattr(pop3, a)[0] = -1
            assert not np.array_equal(getattr(pop3, a), getattr(pop2,a))
        assert np.array_equal(pop2.genmap, pop3.genmap)
        # Test that populations are now independent

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

class TestPopulationLoci:
    """Test methods of Population object relating to obtaining and
    manipulated lists of chromosomes and loci."""

    def test_chrs(self, pop):
        pop2 = pop.clone()
        g,c1,c2 = pop2.genomes, pop2.chrs(0), pop2.chrs(1)
        # Check shapes
        assert c1.shape == (2, pop2.N, pop2.chr_len)
        assert c2.shape == (2, pop2.N, len(pop2.genmap), pop2.n_base)
        # Simple individual/bit version
        assert np.array_equal(g[:,:pop2.chr_len], c1[0])
        assert np.array_equal(g[:,pop2.chr_len:], c1[1])
        # Reshaped individual/locus/bit version
        for locus in xrange(len(pop2.genmap)):
            bits = np.arange(pop2.n_base) + locus*pop2.n_base
            assert np.array_equal(g[:,bits],c2[0,:,locus,:])
            assert np.array_equal(g[:,bits + pop2.chr_len],c2[1,:,locus,:])

    def test_sorted_loci(self, pop):
        pop2 = pop.clone()
        c,l = pop2.chrs(1), pop2.sorted_loci()
        c2 = np.sum(np.sum(c, 3), 0) # Sum over loci and chromosomes
        assert l.shape == c2.shape
        c3 = c2[:,pop2.genmap_argsort] # Sort loci by genmap position
        assert np.array_equal(l, c3)

    def test_loci_subsets(self, pop):
        """Test functionality of surv_loci, repr_loci and neut_loci
        subsetting methods, given correct sorted_loci functionality."""
        def surv_loci_match(p, ref):
            g = np.sort(p.genmap)
            return np.array_equal(p.surv_loci(),ref[:,g<=pop.repr_offset])
        def repr_loci_match(p, ref):
            g = np.sort(p.genmap)
            return np.array_equal(p.repr_loci(),
                    ref[:,np.logical_and(g>= pop.repr_offset,g<pop.neut_offset)])
        def neut_loci_match(p, ref):
            g = np.sort(p.genmap)
            return np.array_equal(p.neut_loci(),ref[:,g>=pop.neut_offset])
        sublocus_tests = [surv_loci_match, repr_loci_match, neut_loci_match]
        pop2 = pop.clone()
        n = pop2.N
        # First test for loci attribute and sorted_loci method
        for s in sublocus_tests:
            assert s(pop2, pop2.loci)
            assert s(pop2, pop2.sorted_loci())
        # Next test relation is preserved upon adding/subtracting members
        pop2.add_members(pop2)
        for s in sublocus_tests:
            assert s(pop2, pop2.loci)
            assert s(pop2, pop2.sorted_loci())
        pop2.subtract_members(np.arange(n))
        for s in sublocus_tests:
            assert s(pop2, pop2.loci)
            assert s(pop2, pop2.sorted_loci())
        # Next test that relationship is broken if members not added to loci
        l = pop2.loci
        for a in attrs_no_loci:
            setattr(pop2, a, np.tile(getattr(pop2, a),
                (2,) + (1,) * (np.ndim(getattr(pop2, a)-1))))
        pop2.N *= 2
        assert np.array_equal(pop2.loci, l)
        for s in sublocus_tests:
            assert s(pop2, pop2.loci)
            assert not s(pop2, pop2.sorted_loci())
        # Finally test that relationship restored by resetting loci attribute
        pop2.loci = pop2.sorted_loci()
        for s in sublocus_tests:
            assert s(pop2, pop2.loci)
            assert s(pop2, pop2.sorted_loci())
