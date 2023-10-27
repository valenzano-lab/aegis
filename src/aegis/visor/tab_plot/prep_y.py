from aegis.help import analyzer


# x-axis is age
def get_total_survivorship(container, iloc=-1):
    return analyzer.get_total_survivorship(container).iloc[iloc]


def get_total_mortality(container, iloc=-1):
    return analyzer.get_total_mortality(container).iloc[iloc]


def get_intrinsic_mortality(container, iloc=-1):
    return analyzer.get_intrinsic_mortality(container).iloc[iloc]


def get_intrinsic_survivorship(container, iloc=-1):
    return analyzer.get_intrinsic_survivorship(container).iloc[iloc]


def get_fertility(container, iloc=-1):
    return analyzer.get_fertility(container).iloc[iloc]


def get_cumulative_reproduction(container, iloc=-1):
    return analyzer.get_cumulative_reproduction(container).iloc[iloc]


def get_birth_structure(container, iloc=-1):
    return analyzer.get_birth_structure(container).iloc[iloc]


def get_death_structure(container, iloc=-1):
    return analyzer.get_death_structure(container).iloc[iloc]


# x-axis is stage
def get_lifetime_reproduction(container):
    return analyzer.get_lifetime_reproduction(container)


def get_life_expectancy(container):
    return analyzer.get_life_expectancy(container)


# x-axis is other
def get_derived_allele_freq(container, iloc=-1):
    return analyzer.get_derived_allele_freq(container).iloc[iloc]


def get_bit_states(container):
    return analyzer.get_sorted_allele_frequencies(container)
