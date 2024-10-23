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
    next_trait_start_position = 0
    for traitname in constants.EVOLVABLE_TRAITS:
        trait = Trait(
            name=traitname,
            cnf=parametermanager.parameters,
            start_position=next_trait_start_position,
            genarch_type=parametermanager.parameters.GENARCH_TYPE,
            MODIF_GENOME_SIZE=parametermanager.parameters.MODIF_GENOME_SIZE,
        )
        traits[traitname] = trait
        next_trait_start_position = trait.end

    self.traits = traits
