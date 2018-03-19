from aegis.Core import chance, init_ages, init_genomes, init_generations
from aegis.Core import quantile, fivenum, init_gentimes, make_windows
import numpy as np
import pytest, random

magnitude = 3
precision = 0.01

class TestFunctionsChance:
    """Tests for functions involved in generating random boolean
    arrays."""

    @pytest.mark.parametrize("p", [0,1])
    def test_chance_degenerate(self, p):
        """Tests wether p=1 returns True/1 and p=0 returns False/0."""
        shape=(10**magnitude, 10**magnitude)
        assert np.all(chance(p, shape).astype(int) == p)

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_chance(self, p):
        """Test that the shape of the output is correct and that the mean
        over many trials is close to the expected value."""
        shape=(random.randint(1,10**magnitude),random.randint(1,10**magnitude))
        c = chance(p, shape)
        s = c.shape
        assert c.shape == shape and c.dtype == "bool"
        assert abs(p-np.mean(c)) < precision

class TestFunctionsInit:
    """Tests for functions involved in simulation initialisation."""

    def test_make_windows(self):
        """Test make_windows for one pair of dummy params."""
        # invalid window size
        with pytest.raises(ValueError):
            make_windows([],0)
        # ws = 1
        tarr = np.linspace(0,1000,5).astype(int)
        assert np.array_equal(tarr, make_windows(tarr,1))
        # general with last=True
        tws = 100
        exp = np.vstack((np.arange(100), np.arange(200,300), np.arange(450,550),\
                np.arange(700,800), np.arange(900,1000)))
        assert np.array_equal(make_windows(tarr,tws), exp)
        assert np.array_equal(make_windows(tarr,tws+1), exp)
        # general with last=False
        exp2 = np.vstack((np.arange(100), np.arange(200,300), np.arange(450,550),\
                np.arange(700,800), np.arange(950,1050)))
        assert np.array_equal(make_windows(tarr, tws, False), exp2)

    def test_pop_inits(self):
        """Test that init_ages, init_genomes and init_generations
        return arrays of the expected dimensions and content."""
        assert np.array_equal(init_ages(), np.array([-1]))
        assert np.array_equal(init_genomes(), np.array([[-1],[-1]]))
        assert np.array_equal(init_generations(), np.array([-1]))
        assert np.array_equal(init_gentimes(), np.array([-1]))

class TestFunctionsFivenum:
    """Tests for functions involved in generating five-number summaries
    of numeric arrays."""

    def test_quantile(self):
        """Test that the quantile() function correctly returns the
        appropriate p-quantile of a simple distribution."""
        a = np.arange(101)
        b,c = -a, a/2.0
        for n in xrange(3):
            p = random.randrange(101)/100.0
            assert np.allclose(quantile(a, p), p*100)
            assert np.allclose(quantile(b, p),  -(100-p*100))
            assert np.allclose(quantile(c, p), p*50)

    def test_fivenum(self):
        a = np.arange(101)
        b,c = -a, a/2.0
        assert np.allclose(fivenum(a), np.arange(0, 101, 25))
        assert np.allclose(fivenum(b), np.arange(-100, 1, 25))
        assert np.allclose(fivenum(c), np.arange(0, 51, 12.5))
