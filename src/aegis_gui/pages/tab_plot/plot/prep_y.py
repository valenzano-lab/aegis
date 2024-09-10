from aegis_sim.utilities.container import Container
from aegis_sim.utilities.analysis import survival, reproduction, genome


# x-axis is age
def get_total_survivorship(container: Container, iloc=-1):
    ys = container.get_surv_observed_interval()
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc].cumprod()
    return ys, max_iloc


def get_mortality_observed(container: Container, iloc=-1):
    ys = container.get_surv_observed_interval()
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc].pipe(lambda x: 1 - x)
    return ys, max_iloc


def get_mortality_intrinsic(container: Container, iloc=-1):
    ys = container.get_phenotype_intrinsic_interval("surv")
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc].pipe(lambda x: 1 - x).to_numpy()
    return ys, max_iloc


def get_intrinsic_survivorship(container: Container, iloc=-1):
    ys = container.get_phenotype_intrinsic_interval("surv")
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc].cumprod()
    return ys, max_iloc


def get_fertility(container: Container, iloc=-1):
    ys = container.get_fert_observed_interval()
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc]
    return ys, max_iloc


def get_cumulative_reproduction(container: Container, iloc=-1):
    ys = reproduction.get_cumulative_reproduction(
        container.get_phenotype_intrinsic_interval("repr"),
        container.get_config()["AGE_LIMIT"],
        container.get_config()["MATURATION_AGE"],
    )
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc]
    return ys, max_iloc


def get_birth_table(container: Container, iloc=-1):
    ys = container.get_birth_table_observed_interval()
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc]
    return ys, max_iloc


def get_death_table(container: Container, iloc=-1):
    ys = container.get_death_table_observed_interval().unstack("cause_of_death")
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc].unstack("cause_of_death")
    return ys, max_iloc


def get_death_table_normalized(container: Container, iloc=-1):
    ys, max_iloc = get_death_table(container=container, iloc=iloc)
    ys = ys.div(ys.sum(1), axis=0)
    return ys, max_iloc


def get_life_table(container: Container, iloc=-1):
    ys = container.get_life_table_observed_interval(normalize=True)
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc]
    return ys, max_iloc


# x-axis is step
def get_lifetime_reproduction(container: Container, iloc=None):
    ys = container.get_lifetime_reproduction()
    max_iloc = None
    return ys, max_iloc


def get_life_expectancy(container: Container, iloc=None):
    ys = survival.get_life_expectancy(
        container.get_phenotype_intrinsic_interval("surv"),
        container.get_config()["AGE_LIMIT"],
    )
    max_iloc = None
    return ys, max_iloc


def get_population_size_after_reproduction(container: Container, iloc=None):
    ys = container.get_population_size_after_reproduction().popsize
    max_iloc = None
    return ys, max_iloc


# x-axis is other
def get_derived_allele_freq(container: Container, iloc=-1):
    ys = genome.get_derived_allele_freq(container.get_genotypes_intrinsic_interval())
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc].to_numpy()
    ys = ys[ys != 0]
    return ys, max_iloc


def get_bit_states(container: Container, iloc=None):
    ys = container.get_genotypes_intrinsic_interval()  # unsorted
    max_iloc = None
    return ys, max_iloc
