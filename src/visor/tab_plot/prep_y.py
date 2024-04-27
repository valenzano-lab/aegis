from aegis.utilities.container import Container
from aegis.utilities.analysis import survival, reproduction, genome


# x-axis is age
def get_total_survivorship(container: Container, iloc=-1):
    return container.get_surv_observed_interval(record_index=iloc).cumprod()


def get_mortality_observed(container: Container, iloc=-1):
    return container.get_surv_observed_interval(record_index=iloc).pipe(lambda x: 1 - x)


def get_mortality_intrinsic(container: Container, iloc=-1):
    return container.get_phenotype_intrinsic_interval("surv").iloc[iloc].pipe(lambda x: 1 - x)


def get_intrinsic_survivorship(container: Container, iloc=-1):
    return container.get_phenotype_intrinsic_interval("surv").iloc[iloc].cumprod()


def get_fertility(container: Container, iloc=-1):
    return container.get_fert_observed_interval(record_index=iloc)
    return reproduction.get_fertility(
        container.get_phenotype_intrinsic_interval("repr"),
        container.get_config()["AGE_LIMIT"],
        container.get_config()["MATURATION_AGE"],
    ).iloc[iloc]


def get_cumulative_reproduction(container: Container, iloc=-1):
    return reproduction.get_cumulative_reproduction(
        container.get_phenotype_intrinsic_interval("repr"),
        container.get_config()["AGE_LIMIT"],
        container.get_config()["MATURATION_AGE"],
    ).iloc[iloc]


def get_birth_structure(container: Container, iloc=-1):
    return container.get_birth_table_observed_interval().iloc[iloc]


def get_causes_of_death(container: Container, iloc=-1):
    age_vs_cause = (
        container.get_death_table_observed_interval().unstack("cause_of_death").iloc[iloc].unstack("cause_of_death")
    )
    return age_vs_cause


# x-axis is stage
def get_lifetime_reproduction(container: Container):
    return container.get_lifetime_reproduction()
    return reproduction.get_lifetime_reproduction(
        container.get_phenotype_intrinsic_interval("surv"),
        container.get_config()["AGE_LIMIT"],
        container.get_config()["MATURATION_AGE"],
    )


def get_life_expectancy(container: Container):
    return survival.get_life_expectancy(
        container.get_phenotype_intrinsic_interval("surv"),
        container.get_config()["AGE_LIMIT"],
    )


# x-axis is other
def get_derived_allele_freq(container: Container, iloc=-1):
    derived_allele_freq = (
        genome.get_derived_allele_freq(container.get_genotypes_intrinsic_interval()).iloc[iloc].to_numpy()
    )
    derived_allele_freq = derived_allele_freq[derived_allele_freq != 0]
    return derived_allele_freq


def get_bit_states(container: Container):
    return container.get_genotypes_intrinsic_interval() # unsorted
    return genome.get_sorted_allele_frequencies(container.get_genotypes_intrinsic_interval()) # sorted
