import numpy as np
import aegis_sim
from aegis_sim.dataclasses.population import Population
import os
import pytest

@pytest.fixture(scope="module")
def setup_aegis():
    test_config = os.path.join(os.path.dirname(__file__), "test_sim.yml")
    aegis_sim.init(test_config, overwrite=True)

@pytest.fixture
def population(setup_aegis):
    return Population.initialize(100, AGE_LIMIT=50)

def test_sample_normal_fraction(population):
    sample_pop = population.sample(0.1)
    assert len(sample_pop) == 10

def test_sample_minimum_one_individual(population):
    tiny_pop = population[:2]
    sample_pop = tiny_pop.sample(0.1)
    assert len(sample_pop) == 1

def test_sample_large_fraction(population):
    sample_pop = population.sample(0.9)
    assert len(sample_pop) == 90

def test_sample_full_population(population):
    sample_pop = population.sample(1.0)
    assert len(sample_pop) == 100

def test_sample_single_individual(population):
    single_pop = population[:1]
    sample_pop = single_pop.sample(0.5)
    assert len(sample_pop) == 1

def test_sample_zero_fraction_raises_error(population):
    with pytest.raises(ValueError, match="Fraction must be greater than 0"):
        population.sample(0)

def test_sample_negative_fraction_raises_error(population):
    with pytest.raises(ValueError, match="Fraction must be greater than 0"):
        population.sample(-0.1)

def test_sample_fraction_greater_than_one_raises_error(population):
    with pytest.raises(ValueError, match="Fraction must be less than or equal to 1"):
        population.sample(1.5)