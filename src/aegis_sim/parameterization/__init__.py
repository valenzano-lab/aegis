from aegis_sim import constants
from aegis_sim.parameterization.parametermanager import ParameterManager

parametermanager = ParameterManager()

traits = None  # will be redefined below in init_traits


def init_traits(self):
    """
    Here the trait order is hardcoded.
    """
    from aegis_sim.parameterization.trait import Trait

    traits = {}
    for traitname in constants.EVOLVABLE_TRAITS:
        trait = Trait(name=traitname, cnf=parametermanager.parameters)
        traits[traitname] = trait

    self.traits = traits
