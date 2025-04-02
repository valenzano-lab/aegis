GENETIC_TRAITS = ("surv", "repr", "muta", "neut", "grow")
TRAIT_N = len(GENETIC_TRAITS)


def starting_site(trait_name):
    return GENETIC_TRAITS.index(trait_name)


VALID_CAUSES_OF_DEATH = (
    "intrinsic",
    "abiotic",
    "infection",
    "predation",
    "starvation",
    "age_limit",
)
