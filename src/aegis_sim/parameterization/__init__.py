import logging
from aegis_sim import constants
from aegis_sim.parameterization.parametermanager import ParameterManager

parametermanager = ParameterManager()

traits = None  # will be redefined below in init_traits
expected_phenotype_length = None  # will be redefined below in init_traits


def init_traits(self):
    """
    Here the trait order is hardcoded.
    """
    from aegis_sim.parameterization.trait import Trait

    traits = {}
    self.expected_phenotype_length = 0

    next_trait_start_position = 0
    for traitname in constants.GENETIC_TRAITS:
        trait = Trait(
            name=traitname,
            cnf=parametermanager.parameters,
            start_position=next_trait_start_position,
            genarch_type=parametermanager.parameters.GENARCH_TYPE,
            MODIF_GENOME_SIZE=parametermanager.parameters.MODIF_GENOME_SIZE,
        )
        traits[traitname] = trait
        next_trait_start_position = trait.end

        if trait.evolvable:
            if trait.agespecific:
                self.expected_phenotype_length += parametermanager.parameters.AGE_LIMIT
            else:
                self.expected_phenotype_length += 1

    logging.info(f"Expected phenotype length is {self.expected_phenotype_length}")
    self.traits = traits
