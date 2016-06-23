# test module for functions in gs_functions
# configuration, recording and output functions are not tested

from gs_functions import *
import pytest
import numpy

# random number generation
@pytest.mark.parametrize("arg1, arg2", [(0,1), (1,1), (0.5,(1000,1000))])
def test_chance(arg1, arg2):
    """Tests wether P=1 returns True, P=0 returns 0 and if the deviation for
    the values between boundries is less than 0.1%."""
    ans = np.mean(chance(arg1, arg2).astype(int))
    assert ans == arg1 or (ans > arg1-0.001 and ans < arg1+0.001)

# population generation
@pytest.fixture
def conf(request):
    return get_conf("config")

def test_make_genome_array(conf):
    """Tests only whether genome array has the right shape based on values
    defined in config.py file."""
    assert make_genome_array(conf.start_pop,conf.chr_len,conf.gen_map,conf.n_base,conf.g_dist).shape == (conf.start_pop,conf.chr_len*2)

# update functions
@pytest.mark.parametrize("res0,N", [(1000,500),(4000,500),(0,1500)])
def test_update_resources(res0,N):
    """Test if update_resources return a value within the definition interval for
    three possible scenarios: shortage [=0], abundance [>0], upper limit."""
    ans = update_resources(res0,N,1000,1.6,5000)
    assert (ans >= 0) and (ans <= 5000)
