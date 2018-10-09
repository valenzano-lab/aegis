from aegis.Core import Config, Population # Classes
from aegis.Core import chance, init_ages, init_genomes, init_generations
from aegis.Core import init_gentimes
import pytest, random, copy
import numpy as np

from test_1_Config import conf_path, gen_trseed

@pytest.fixture(scope="module")
def conf(request, conf_path):
    c = Config(conf_path)
    c["n_stages"] = "auto"
    c["scale"] = 1.01
    alpha = c["m_rate"]
    beta = c["m_rate"]*c["m_ratio"]
    c["mu"] = beta/(alpha+beta) # for 1's
    # optimise parameters for testing
    delta = 0.01*c["mu"] # choose delta
    epsbar = delta  # choose epsbar
    c["zeta"] = 0.03     # choose zeta
    c["eps"] = epsbar + delta
    gsize = (2*c["max_ls"]-c["maturity"]+c["n_neutral"])*c["n_base"]*2
    print gsize
    samplesize = 1.0/(2*epsbar**2)*np.log(2.0/c["zeta"]) # compute sample size
    print samplesize
    c["start_pop"] = (int(samplesize/gsize)+2)/2*2
    print c["start_pop"]
    c.generate()
    return c

@pytest.fixture(scope="module")
def pop(request, conf):
    """Create a sample population from the default configuration."""
    return Population(conf["params"], conf["genmap"], conf["mapping"], init_ages(),
            init_genomes(), init_generations(), init_gentimes())

@pytest.mark.skip
def test_autostage_asex(conf, pop):
    """
    Test that without selection, genomes of an only mutating population
    fall within the predicted precision of the expected value.
    """
    print "auto: ", conf["auto"]
    c = conf.copy()
    p = pop.clone()

    snaps = (np.linspace(0,c["min_gen"],100)).astype(int)
    obs = []
    for i in range(c["min_gen"]):
        if i in snaps: obs.append(p.genomes.mean())
        p.mutate(c["m_rate"], c["m_ratio"])
    print "min gen: ", c["min_gen"]
    print "pop size: ", pop.N
    print "mu: ", c["mu"]
    print "epsilon: ", c["eps"]
    print "abs(obs-mu)<eps:"
    for x,y in zip(obs,snaps): print y,":\t",abs(x-c["mu"])<c["eps"],"\t", (x-c["mu"])
    print "zeta: ", c["zeta"]
    assert np.isclose(c["mu"], obs[-1], atol=c["eps"])

@pytest.mark.skip
def test_autostage_sex(conf, pop):
    """
    Test that without selection, genomes of a mutating, recombining and
    assorting population fall within the predicted precision of the expected value.
    """
    print "auto: ", conf["auto"]
    c = conf.copy()
    p = pop.clone()

    snaps = (np.linspace(0,c["min_gen"],200)).astype(int)
    obs = []
    for i in range(c["min_gen"]):
        if i in snaps: obs.append(p.genomes.mean())
        # double the population since assortment halfs it
        p2 = p.clone()
        for population in (p,p2):
            # recombination
            #population.recombination(c["r_rate"])
            # assortment
            #population.shuffle()
            parent_0 = np.arange(population.N/2)*2      # Parent 0
            parent_1 = parent_0 + 1                     # Parent 1
            # Chromosome from parent 0
            which_chr_0 = chance(0.5,population.N/2, population.prng)*1
            # Chromosome from parent 1
            which_chr_1 = chance(0.5,population.N/2, population.prng)*1
            chrs = np.copy(population.chrs(False))
            population.genomes[::2,:population.chr_len] = chrs[which_chr_0, parent_0]
            population.genomes[::2,population.chr_len:] = chrs[which_chr_1, parent_1]
            population.subset_members(np.tile([True,False], population.N/2))
            # mutation
            p.mutate(c["m_rate"], c["m_ratio"])
        p.add_members(p2)
    print "min gen: ", c["min_gen"]
    print "pop size: ", pop.N
    print "mu: ", c["mu"]
    print "epsilon: ", c["eps"]
    print "abs(obs-mu)<eps:"
    for x,y in zip(obs,snaps): print y,":\t",abs(x-c["mu"])<c["eps"],"\t", (x-c["mu"])
    print "zeta: ", c["zeta"]
    assert np.isclose(c["mu"], obs[-1], atol=c["eps"])
