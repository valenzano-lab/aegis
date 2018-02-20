from aegis.Core import Config, Population # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
import pytest, random, copy
import numpy as np

# Import fixtures
from test_1_Config import conf, conf_path, ran_str, gen_trseed
from test_2a_Population_init import pop

precision = 0.02

class TestPopulationSubpop:
    """Tests of Population subsetting using get_subpop."""

    def test_get_subpop_degen_probs(self, pop):
        """Confirm that get_subpop returns all individuals when
        selection probability is 1 and none when selection probability
        is 0."""
        p = pop.clone()
        age_range = np.array([p.maturity, p.max_ls])
        p.ages = np.random.randint(age_range[0], age_range[1], p.N)
        which_loci = np.random.randint(0, len(p.genmap), np.diff(age_range))
        ab, vr = age_range, np.linspace(0,1,2*p.n_base+1)
        # 1: Selection probability of 1
        p.genomes[:,:] = 1
        p.loci = p.sorted_loci()
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.sum(gs) == p.N
        # 2: Selection probability of 0
        p.genomes[:,:] = 0
        p.loci = p.sorted_loci()
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.sum(gs) == 0

    def test_get_subpop_probs_iid(self, pop):
        """Test that the probability of passing is close to that indicated
        by the genome (when all loci have the same distribution."""
        # Setup
        p = pop.clone()
        p.add_members(p)
        p.add_members(p)
        age_range = np.array([0, p.max_ls])
        p.ages = np.random.randint(age_range[0], age_range[1], p.N)
        which_loci = np.random.randint(0, len(p.genmap), np.diff(age_range))
        ab, vr = age_range, np.linspace(0,1,2*p.n_base+1)
        x = random.random()
        # Iid Bernoulli-distributed bits in all genomes
        p.genomes = chance(x, p.genomes.shape).astype(int)
        p.loci = p.sorted_loci()
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.isclose(np.mean(gs), x, atol=precision*2)

    def test_get_subpop_exclude_ages(self, pop):
        """Confirm that individuals outside the specified age range are never
        returned by get_subpop."""
        # Set up age range and population
        p = pop.clone()
        min_age = random.randrange(p.max_ls-6)
        max_age = random.randrange(min_age+5, p.max_ls-1)
        print min_age, max_age
        age_range = np.array([min_age, max_age])
        which_loci = np.random.randint(0, len(p.genmap), np.diff(age_range))
        ab, vr = age_range, np.linspace(0,1,2*p.n_base+1)
        p.genomes[:,:] = 1 # Selection probability of 1 if included
        p.loci = p.sorted_loci()
        # 1: All too young
        p.ages[:] = min_age - 1
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.sum(gs) == 0
        # 2: All too old
        p.ages[:] = max_age + 1
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.sum(gs) == 0
        # 3: All right age
        p.ages[:] = max_age - 1
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.sum(gs) == p.N
        # 4: Some in, some out
        p.add_members(p)
        p.add_members(p) # Make population bigger
        x = random.random()
        p.ages[:] = min_age - 1 + chance(x, p.ages.shape)*2
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.isclose(np.mean(gs), x, atol=precision*2)

    def test_get_subpop_different(self, pop):
        """Test whether individuals with different genotypes are selected
        with appropriately different frequencies by get_subpop."""
        # Set up population with different genotypes
        p = pop.clone()
        p.add_members(p)
        p.add_members(p)
        p.ages[:] = p.maturity+1
        x,y = int(p.N / 3), random.random()
        p.genomes[:x,:] = 0
        p.genomes[x:2*x,:] = 1
        p.genomes[2*x:,:] = chance(y,p.genomes[2*x:,:].shape)
        p.loci = p.sorted_loci()
        # Calculate subpop
        age_range = np.array([0,p.max_ls])
        which_loci = np.random.randint(0, len(p.genmap), np.diff(age_range))
        ab, vr = age_range, np.linspace(0,1,2*p.n_base+1)
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.mean(gs[:x]) == 0
        assert np.mean(gs[x:2*x]) == 1
        assert np.isclose(np.mean(gs[2*x:]), y, atol=precision*2)

    def test_subpop_extreme_probs(self, pop):
        """Test that get_subpop appropriately handles 'probabilities'
        outside of the range [0,1] (e.g. as the result of starvation."""
        p = pop.clone()
        age_range = np.array([p.maturity, p.max_ls])
        p.ages = np.random.randint(age_range[0], age_range[1], p.N)
        which_loci = np.random.randint(0, len(p.genmap), np.diff(age_range))
        n = random.randint(2,10)
        ab,vr = age_range, np.linspace(-n, n, 2 * p.n_base + 1)
        # 1: Selection probability > 1 -> probability == 1
        p.genomes[:,:] = 1
        p.loci = p.sorted_loci()
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.sum(gs) == p.N
        # 2: Selection probability < 0 -> probability == 0
        p.genomes[:,:] = 0
        p.loci = p.sorted_loci()
        gs = p.get_subpop(ab, p.sorted_loci()[:,which_loci], vr)
        assert np.sum(gs) == 0

        #! TODO: Test functionality for p_ranges other than 0->1

class TestPopulationDeath:
    """Test methods relating to death, given that get_subpop was
    already tested above."""

    def test_death(self, pop):
        """Test if self.death() correctly inverts death probabilities
        and incorporates starvation factor to get survivor probabilities
        and survivor array."""
        # Generate and validate test population
        p,x = pop.clone(), random.random()
        p.add_members(p)
        p.add_members(p)
        assert len(p.genomes) == len(p.loci) == len(p.ages)
        pmin, pmax = 0.8, 0.99
        n,surv_r = float(p.N), np.linspace(pmin, pmax, 2*pop.n_base+1)
        starvation = random.uniform(2,5)
        # Modify survival loci to be uniform with mean of x
        #p.genomes[:,:] = chance(x, p.genomes.shape).astype(int)
        b = p.n_base
        surv_loci = np.nonzero(p.genmap<p.repr_offset)[0]
        surv_pos = np.array([range(p.n_base) + y for y in surv_loci*p.n_base])
        surv_pos = np.append(surv_pos, surv_pos + p.chr_len)
        p.genomes[:, surv_pos] =\
                chance(x, (p.N, len(surv_pos))).astype(int)
        p.loci = p.sorted_loci() # Update loci for new genomes
        # Call and test death function, with and without starvation
        p2 = p.clone()
        p.death(surv_r, 1)
        p2.death(surv_r, starvation)
        # Define observation and expectation in terms of possible range
        s1 = pmin + (pmax-pmin)*x
        s2 = 1-(1-s1)*starvation
        print p.N/n, s1, x, p.N/n - s1
        assert np.isclose(p.N/n, s1, atol=precision)
        print p2.N/n, s2, p2.N/n - s2
        assert np.isclose(p2.N/n, s2, atol=precision)

    def test_death_extreme_starvation(self, pop):
        """Confirm that death() handles extreme starvation factors
        correctly (probability limits at 0 and 1)."""
        p0 = pop.clone()
        p1 = pop.clone()
        p2 = pop.clone()
        s_range = np.linspace(0, 0.99, 2*pop.n_base+1)
        p0.death(s_range, 1e10)
        p1.death(s_range, -1e10)
        p2.death(s_range, 1e-10)
        assert p0.N == 0
        assert p1.N == pop.N
        assert p2.N == pop.N

class TestPopulationGrowth:
    """Test methods relating to reproduction, given that get_subpop was
    already tested above."""

    # 1: Mutation (all reproductive modes)

    def test_mutate_degen(self, pop):
        """Test that no mutation occurs if mrate = 0 and total mutation
        occurs if mrate = 1."""
        p1,p2 = pop.clone(), pop.clone()
        p1.mutate(0,1)
        p2.mutate(1,1)
        assert np.mean(pop.genomes != p1.genomes) == 0
        assert np.mean(pop.genomes != p2.genomes) == 1

    def test_mutate_unbiased(self, pop):
        """Test that, in the absence of a +/- bias, the appropriate
        proportion of the genome is mutated."""
        p, mrate = pop.clone(), random.random()
        genomes = np.copy(p.genomes)
        p.mutate(mrate,1)
        assert np.isclose(np.mean(genomes != p.genomes),mrate,atol=precision/2)

    def test_mutate_biased(self, pop):
        """Test that the bias between positive and negative mutations is
        implemented correctly."""
        p, mrate, mratio = pop.clone(), 0.5, random.random()
        g0 = np.copy(p.genomes)
        p.mutate(mrate, mratio)
        g1 = p.genomes
        is1 = (g0==1)
        is0 = np.logical_not(is1)
        assert np.isclose(np.mean(g0[is1] != g1[is1]), mrate, atol=precision/2)
        assert np.isclose(np.mean(g0[is0] != g1[is0]), mrate*mratio,
                atol=precision/2)

    def test_mutate_loci(self, pop):
        """Test that loci update correctly following genome mutation."""
        p, mrate, mratio = pop.clone(), random.random(), random.random()
        assert np.array_equal(p.loci, p.sorted_loci())
        p.mutate(mrate, mratio)
        assert np.array_equal(p.loci, p.sorted_loci())
        assert not np.array_equal(p.loci, pop.sorted_loci())

    # 2: Recombination (recombine_only, sexual)

    def test_recombine_degen(self, pop):
        """Test that no recombination occurs when rrate = 0 and total
        recombination occurs when rrate = 1."""
        p = pop.clone()
        # Set chromosome 1 to 0's and chromosome 2 to 1's
        p.genomes[:,:p.chr_len] = 0
        p.genomes[:,p.chr_len:] = 1
        p.loci = p.sorted_loci()
        # Condition 1: rrate = 0, chromosomes unchanged
        p1 = p.clone()
        p1.recombination(0.0)
        assert np.array_equal(p1.chrs(0)[0], np.zeros([p1.N, p1.chr_len]))
        assert np.array_equal(p1.chrs(0)[1], np.ones([p1.N, p1.chr_len]))
        # Condition 2: rrate = 1, chromosomes interleaved
        p2 = p.clone()
        p2.recombination(1.0)
        exp_chr_0 = np.tile([1,0], [p2.N, p2.chr_len/2])
        exp_chr_1 = np.tile([0,1], [p2.N, p2.chr_len/2])
        if p2.chr_len % 2 == 1:
            exp_chr_0 = np.concatenate((exp_chr_0, np.ones([p2.N, 1])), 1)
            exp_chr_1 = np.concatenate((exp_chr_1, np.zeros([p2.N, 1])), 1)
        assert np.array_equal(p2.chrs(0)[0], exp_chr_0)
        assert np.array_equal(p2.chrs(0)[1], exp_chr_1)

    def test_recombine_random(self, pop):
        """Test that the correct proportion of bits are recombined when
        0 < rrate < 1."""
        p, rrate = pop.clone(), random.random()
        # Set chromosome 1 to 0's and chromosome 2 to 1's
        p.genomes[:,:p.chr_len] = 0
        p.genomes[:,p.chr_len:] = 1
        p.loci = p.sorted_loci()
        p.recombination(rrate)
        # Compute expected mean value of chromosome 1 (somewhat complicated)
        m, s = float(p.chr_len), 1-2*rrate
        M = 0.5 - (s*(1-s**m))/(2*m*(1-s))
        assert np.isclose(np.mean(p.genomes[:,:p.chr_len]), M,
                atol=precision)
        assert np.isclose(np.mean(p.genomes[:,p.chr_len:]), 1-M,
                atol=precision)

    # 3: Assortment (assort_only, sexual)

    def test_assortment_solo(self, pop):
        """Test that assortment correctly raises an error when given a
        single parent."""
        p = pop.clone()
        p.shuffle()
        p.subtract_members(xrange(1,p.N)) # Leave only one parent
        with pytest.raises(ValueError):
            p.assortment()

    def test_assortment_pair(self, pop):
        """Test that assortment correctly assorts the chromosomes of
        two parents into a single child."""
        p = pop.clone()
        p.shuffle()
        p.subtract_members(xrange(2,p.N)) # Two random parents
        p_chrs = np.vstack(p.chrs(0).transpose((1,0,2)))
        p.assortment()
        assert p.N == 1
        c_chrs = np.vstack(p.chrs(0).transpose((1,0,2)))
        # Identify parent of each chromosome
        ident = np.array([[np.array_equal(x,y)\
                for y in p_chrs] for x in c_chrs])
        parent = np.nonzero(ident)[1]/2
        # Each chromosome should be from a different parent
        assert np.array_equal(np.unique(parent), np.sort(parent))

    def test_assortment_odd(self, pop):
        """Test that assortment correctly discards a random member to
        guarantee an even number of parents."""
        p = pop.clone()
        p.shuffle()
        p.subtract_members(xrange(5,p.N)) # Five random parents
        p_chrs = np.vstack(p.chrs(0).transpose((1,0,2)))
        p.assortment()
        assert p.N == 2
        c_chrs = np.vstack(p.chrs(0).transpose((1,0,2)))
        # Identify parent of each chromosome
        ident = np.array([[np.array_equal(x,y)\
                for y in p_chrs] for x in c_chrs])
        parent = np.nonzero(ident)[1]/2
        # All four child chromosomes from a different parent
        assert np.array_equal(np.unique(parent), np.sort(parent))

    def test_assortment_genstats_pair(self, pop):
        """Test that child generations and gentimes are correctly
        computed for a single assorted parental pair."""
        p,n = pop.clone(), 2
        p.shuffle()
        p.ages = np.random.randint(0, p.max_ls, p.N)
        p.generations = np.random.randint(0, 100, p.N)
        p.subtract_members(xrange(n,p.N)) # n random parents
        # Make a copy, assort and test
        p2 = p.clone()
        p2.assortment()
        assert p2.ages == np.mean(p.ages).astype(int)
        assert p2.generations == np.max(p.generations)

    def test_assortment_gentimes(self, pop):
        """Test that child generations and gentimes are within expected
        parameters for a larger assorted parental group."""
        # Generate populations with random ages and n pairs
        p,n = pop.clone(), 10
        p.shuffle()
        p.ages = np.random.randint(0, p.max_ls, p.N)
        p.generations = np.random.randint(0, 100, p.N)
        p.subtract_members(xrange(n*2,p.N)) # 2n random parents in n pairs
        # Make a copy and assort
        p2 = p.clone()
        p2.assortment()
        # Test that age sums match
        s1, s2, n2 = np.sum(p.ages), np.sum(p2.ages), p2.N
        print s1, 2*s2, s1-2*s2, n2
        assert s1 - 2*s2 < n2
        # Test that generation
        g, outsum = np.sort(p.generations), np.sum(p2.generations)
        minsum,maxsum = np.sum(g[1::2]), np.sum(g[n:])
        assert minsum <= outsum and outsum <= maxsum

    # 4: Make_children (all modes)
    def test_make_children_child_ages_generations(self, pop):
        """Test that individuals produced by make_children are all of
        age 0 and generation 1, for all reproductive modes."""
        p, rr = pop.clone(), np.linspace(0, 1, pop.n_base*2+1)
        p.generations[:] = 0
        for mode in ["sexual", "asexual", "recombine_only", "assort_only"]:
            p.repr_mode = mode
            p.set_attributes(p.params())
            c = p.make_children(rr, 1, 0, 1, 0)
            assert np.array_equal(c.ages, np.zeros(c.N))
            assert np.array_equal(c.generations, np.ones(c.N))

    def test_make_children_gentimes_noassort(self, pop):
        """Test that individuals produced by make_children all have
        generation times matching parental ages, in the absence of
        assortment."""
        p, rr = pop.clone(), np.linspace(1, 1, pop.n_base*2+1)
        p.generations[:] = 0
        assert np.array_equal(np.tile(0, p.N), p.gentimes)
        for mode in ["asexual", "recombine_only"]:
            print mode
            p.repr_mode = mode
            p.set_attributes(p.params())
            c = p.make_children(rr, 1, 0, 1, 0)
            assert np.array_equal(np.sort(p.ages[p.ages >= p.maturity]),
                    np.sort(c.gentimes))
        # See assortment tests for test of correct computation in sexual case

    def test_make_children_starvation(self, pop):
        """Test if make_children correctly incorporates starvation
        factors to get parentage probability."""
        # Define parameters and expected output
        p,x,y = pop.clone(), 1.0, 3
        print x
        for n in xrange(y): # Increase population by 2**y
            p.add_members(p)
        pmin, pmax = 0.01, 0.2
        n = float(np.sum(p.ages >= p.maturity)) # of adults
        repr_r = np.linspace(pmin, pmax, 2*pop.n_base+1)
        starvation = random.uniform(2,5)
        s1 = pmin + (pmax-pmin)*x
        s2 = s1/starvation
        # Modify reproductive loci to be uniform with mean of x
        repr_loci = np.nonzero(np.logical_and(p.genmap>=p.repr_offset,
                p.genmap<p.neut_offset))[0]
        repr_pos = np.array([range(p.n_base) + y for y in repr_loci*p.n_base])
        repr_pos = np.append(repr_pos, repr_pos + p.chr_len)
        p.genomes[:, repr_pos] =\
                chance(x, (p.N, len(repr_pos))).astype(int)
        p.loci = p.sorted_loci()
        # Call and test death function, with and without starvation
        for mode in ["sexual", "asexual", "recombine_only", "assort_only"]:
            p.repr_mode = mode
            p.set_attributes(p.params())
            p2 = p.clone()
            c = p.make_children(repr_r, 1, 0, 1, 0)
            c2 = p2.make_children(repr_r, starvation, 0, 1, 0)
            exp1 = s1 / (2.0 if mode in ["sexual", "assort_only"] else 1.0)
            exp2 = s2 / (2.0 if mode in ["sexual", "assort_only"] else 1.0)
            assert np.array_equal(c.ages, np.zeros(c.N))
            print mode, c.N/n
            assert np.isclose(c.N/n, exp1, atol=precision*2)
            print mode, c2.N/n
            assert np.isclose(c2.N/n, exp2, atol=precision*2)

    def test_make_children_solo(self, pop):
        """Test that make_children correctly does nothing when given
        exactly one parent, under assorting reproductive modes, and
        returns exactly one child otherwise."""
        p = pop.clone()
        p.shuffle()
        p.subtract_members(xrange(1,p.N)) # Leave one parent
        p.ages[:] = p.maturity # make sure it's an adult
        assert p.N == 1
        r_range = np.linspace(1,1, 2*pop.n_base+1)
        modes = ["sexual", "asexual", "recombine_only", "assort_only"]
        exp = [0, 1, 1, 0]
        for n in xrange(len(modes)):
            print modes[n]
            p.repr_mode = modes[n]
            p.set_attributes(p.params())
            assert p.repr_mode == modes[n]
            c = p.make_children(r_range, 1, 0, 1, 0)
            assert c.N == exp[n]

    def test_make_children_zero(self, pop):
        """Test that make_children correctly does nothing when given an
        empty population of parents."""
        p = pop.clone()
        p.subtract_members(xrange(p.N)) # Leave no parents
        assert p.N == 0
        r_range = np.linspace(1,1, 2*pop.n_base+1)
        modes = ["sexual", "asexual", "recombine_only", "assort_only"]
        for n in xrange(len(modes)):
            p.repr_mode = modes[n]
            p.set_attributes(p.params())
            assert p.repr_mode == modes[n]
            c = p.make_children(r_range, 1, 0, 1, 0)
            assert c.N == 0

    def test_make_children_extreme_starvation(self, pop):
        """Confirm that death() handles extreme starvation factors
        correctly (probability limits at 0 and 1)."""
        r_range = np.linspace(0, 1, 2*pop.n_base+1)
        p = pop.clone()
        n = np.sum(p.ages >= p.maturity)
        for mode in ["sexual", "asexual", "recombine_only", "assort_only"]:
            p.repr_mode = mode
            p.set_attributes(p.params())
            c0 = p.make_children(r_range, 1e10, 0, 1, 0)
            c1 = p.make_children(r_range, -1e10, 0, 1, 0)
            c2 = p.make_children(r_range, 1e-10, 0, 1, 0)
            assert c0.N == 0
            assert c1.N == 0
            assert c2.N-n/(2 if mode in ["sexual", "assort_only"] else 1) <= 1

    def test_growth(self, pop):
        """Test that growth is equivalent to making children and adding
        them to the population."""
        p = pop.clone()
        p.genomes[:,:] = 1
        p.loci = p.sorted_loci()
        r_range = np.linspace(0,1,2*p.n_base+1)
        c = p.make_children(r_range, 1, 0, 1, 0)
        p.growth(r_range, 1, 0, 1, 0)
        assert p.N == pop.N + c.N
        # TODO: Do this with non-degenerate reproduction probability?:
