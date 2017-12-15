from aegis.Core import chance, init_ages, init_genomes, init_generations
import numpy as np
import pytest, random

class TestFunctions:

    @pytest.mark.parametrize("p", [0,1])
    def test_chance_degenerate(self, p):
        """Tests wether p=1 returns True/1 and p=0 returns False/0."""
        shape=(1000,1000)
        assert np.all(chance(p, shape).astype(int) == p)

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_chance(self, p):
        precision = 0.01
        """Test that the shape of the output is correct and that the mean
        over many trials is close to the expected value."""
        shape=(random.randint(1,1000),random.randint(1,1000))
        c = chance(p, shape)
        s = c.shape
        assert c.shape == shape and c.dtype == "bool"
        assert abs(p-np.mean(c)) < precision

    def test_inits(self):
        """Test that init_ages, init_genomes and init_generations
        return arrays of the expected dimensions and content."""
        assert np.array_equal(init_ages(), np.array([-1]))
        assert np.array_equal(init_genomes(), np.array([[-1],[-1]]))
        assert np.array_equal(init_generations(), np.array([-1]))

#! TODO: Test time functions
