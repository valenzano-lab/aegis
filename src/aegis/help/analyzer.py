import pandas as pd
from aegis.panconfiguration import pan


# ANALYZE FEATHER SNAPSHOTS
def get_total_mortality(container):
    total_survivorship = get_total_survivorship(container)
    total_mortality = -total_survivorship.pct_change(axis=1)
    return total_mortality


def get_total_survivorship(container):
    # additive age structure records viability of each individual during its lifetime
    # when applied to a discrete cohort, additive age structure have the same shape as the survivorship curve
    # furthermore, if normalized, it is equivalent to the survivorship curve
    # when not applied to a cohort, the same holds if the population is stationary which approximately holds at short time scales
    additive_age_structure = container.get_df("additive_age_structure")
    total_survivorship = additive_age_structure.div(
        additive_age_structure.iloc[:, 0], axis=0
    )  # normalize
    return total_survivorship


def get_birth_structure(container):
    age_at_birth = container.get_df("age_at_birth")
    y = age_at_birth.iloc[-1]
    y = age_at_birth.div(age_at_birth.sum(1), axis=0)
    return y


def get_death_structure(container, targetcause="genetic"):
    age_at = {
        causeofdeath: container.get_df(f"age_at_{causeofdeath}")
        for causeofdeath in pan.causeofdeath_valid
    }

    pseudocount = 0

    age_at_target = age_at[targetcause] + pseudocount
    age_at_all = pd.DataFrame(
        pseudocount, columns=age_at_target.columns, index=age_at_target.index
    )

    for v in age_at.values():
        age_at_all += v

    return age_at_target / age_at_all


# ANALYZE PICKLED POPULATIONS


# ANALYZE CSVs
# -- Genotypic
def get_sorted_allele_frequencies(container):
    genotypes = container.get_df("genotypes")
    total_frequency = genotypes.sum(0)
    return (
        genotypes.T.assign(total=total_frequency)
        .sort_values(by="total", ascending=False)
        .T.iloc[:-1]
    )


def get_derived_allele_freq(container):
    genotypes = container.get_df("genotypes")
    reference = genotypes.round()
    derived_allele_freq = (
        genotypes.iloc[1:].reset_index(drop=True)
        - reference.iloc[:-1].reset_index(drop=True)
    ).abs()
    return derived_allele_freq


def get_mean_allele_freq(container):
    genotypes = container.get_df("genotypes")
    mean_allele_freq = genotypes.mean(0)
    return mean_allele_freq


def get_quantile_allele_freq(container, quantile):
    genotypes = container.get_df("genotypes")
    quantile_allele_freq = genotypes.quantile(quantile)
    return quantile_allele_freq


# -- Phenotypic
def get_life_expectancy(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    # TODO Ensure that you are slicing the phenotype array at right places
    pdf = phenotypes.iloc[:, :max_age]
    survivorship = pdf.cumprod(1)
    y = survivorship.sum(1)
    return y


def get_intrinsic_mortality(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    # TODO Ensure that you are slicing the phenotype array at right places
    pdf = phenotypes.iloc[:, :max_age]
    y = 1 - pdf
    return y


def get_intrinsic_survivorship(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    # TODO Ensure that you are slicing the phenotype array at right places
    pdf = phenotypes.iloc[:, :max_age]
    y = pdf.cumprod(axis=1)
    return y


def get_fertility_potential(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    # TODO Ensure that you are slicing the phenotype array at right places
    # TODO Ensure that fertility is 0 before maturity
    fertility = phenotypes.iloc[:, max_age:]
    y = fertility
    return y


def get_fertility(container):
    fertility_potential = get_fertility_potential(container)
    maturation_age = container.get_config()["MATURATION_AGE"]
    fertility_potential = fertility_potential.T.reset_index(drop=True).T
    fertility_potential.loc[:, range(0, maturation_age)] = 0
    y = fertility_potential
    return y


def get_cumulative_reproduction(container):
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13289
    # BUG fertility is 0 before maturity
    survivorship = get_intrinsic_survivorship(container).T.reset_index(drop=True).T
    fertility = get_fertility(container).T.reset_index(drop=True).T
    y = (survivorship * fertility).cumsum(axis=1)
    return y


def get_lifetime_reproduction(container):
    # BUG fertility is 0 before maturity
    survivorship = get_intrinsic_survivorship(container).T.reset_index(drop=True).T
    fertility = get_fertility(container).T.reset_index(drop=True).T
    # y = np.sum((np.array(survivorship) * np.array(fertility)), axis=1)
    y = (survivorship * fertility).sum(axis=1)
    return y
