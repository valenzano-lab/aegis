from aegis.utilities.analysis import survival, reproduction, genome


# x-axis is age
def get_total_survivorship(container, iloc=-1):
    return survival.get_total_survivorship(container.get_interval_life_table()).iloc[iloc]


def get_total_mortality(container, iloc=-1):
    return survival.get_total_mortality(container.get_interval_life_table()).iloc[iloc]


def get_intrinsic_mortality(container, iloc=-1):
    return survival.get_intrinsic_mortality(
        container.get_interval_phenotypes(), container.get_config()["AGE_LIMIT"]
    ).iloc[iloc]


def get_intrinsic_survivorship(container, iloc=-1):
    return survival.get_intrinsic_survivorship(
        container.get_interval_phenotypes(), container.get_config()["AGE_LIMIT"]
    ).iloc[iloc]


def get_fertility(container, iloc=-1):
    return reproduction.get_fertility(
        container.get_interval_phenotypes(),
        container.get_config()["AGE_LIMIT"],
        container.get_config()["MATURATION_AGE"],
    ).iloc[iloc]


def get_cumulative_reproduction(container, iloc=-1):
    return reproduction.get_cumulative_reproduction(
        container.get_interval_phenotypes(),
        container.get_config()["AGE_LIMIT"],
        container.get_config()["MATURATION_AGE"],
    ).iloc[iloc]


def get_birth_structure(container, iloc=-1):
    return container.get_interval_birth_table().iloc[iloc]


def get_causes_of_death(container, iloc=-1):
    age_vs_cause = container.get_interval_death_table().unstack("cause_of_death").iloc[iloc].unstack("cause_of_death")
    return age_vs_cause


# x-axis is stage
def get_lifetime_reproduction(container):
    return reproduction.get_lifetime_reproduction(
        container.get_interval_phenotypes(),
        container.get_config()["AGE_LIMIT"],
        container.get_config()["MATURATION_AGE"],
    )


def get_life_expectancy(container):
    return survival.get_life_expectancy(
        container.get_interval_phenotypes(),
        container.get_config()["AGE_LIMIT"],
    )


# x-axis is other
def get_derived_allele_freq(container, iloc=-1):
    derived_allele_freq = genome.get_derived_allele_freq(container.get_interval_genotypes()).iloc[iloc].to_numpy()
    derived_allele_freq = derived_allele_freq[derived_allele_freq != 0]
    return derived_allele_freq


def get_bit_states(container):
    return genome.get_sorted_allele_frequencies(container.get_interval_genotypes())
