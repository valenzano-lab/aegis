EVOLVABLE_TRAITS = ("surv", "repr", "muta", "neut", "grow")
# EVOLVABLE_TRAITS = ("surv", "repr", "muta", "grow")

TRAIT_N = len(EVOLVABLE_TRAITS)


def starting_site(trait_name):
    return EVOLVABLE_TRAITS.index(trait_name)

VALID_CAUSES_OF_DEATH = (
    "intrinsic",
    "abiotic",
    "infection",
    "predation",
    "starvation",
    "age_limit",
)
