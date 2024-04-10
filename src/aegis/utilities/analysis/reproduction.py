from aegis.utilities.analysis.survival import get_intrinsic_survivorship


def get_fertility_potential(interval_phenotypes, AGE_LIMIT):
    # TODO Ensure that you are slicing the phenotype array at right places
    # TODO Ensure that fertility is 0 before maturity
    fertility = interval_phenotypes.iloc[:, AGE_LIMIT:]
    y = fertility
    return y


def get_fertility(interval_phenotypes, AGE_LIMIT, MATURATION_AGE):
    fertility_potential = get_fertility_potential(interval_phenotypes, AGE_LIMIT)
    fertility_potential = fertility_potential.T.reset_index(drop=True).T
    fertility_potential.loc[:, range(0, MATURATION_AGE)] = 0
    y = fertility_potential
    return y


def get_cumulative_reproduction(interval_phenotypes, AGE_LIMIT, MATURATION_AGE):
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13289
    # BUG fertility is 0 before maturity
    survivorship = get_intrinsic_survivorship(interval_phenotypes, AGE_LIMIT).T.reset_index(drop=True).T
    fertility = get_fertility(interval_phenotypes, AGE_LIMIT, MATURATION_AGE).T.reset_index(drop=True).T
    y = (survivorship * fertility).cumsum(axis=1)
    return y


def get_lifetime_reproduction(interval_phenotypes, AGE_LIMIT, MATURATION_AGE):
    # BUG fertility is 0 before maturity
    survivorship = get_intrinsic_survivorship(interval_phenotypes, AGE_LIMIT).T.reset_index(drop=True).T
    fertility = get_fertility(interval_phenotypes, AGE_LIMIT, MATURATION_AGE).T.reset_index(drop=True).T
    # y = np.sum((np.array(survivorship) * np.array(fertility)), axis=1)
    y = (survivorship * fertility).sum(axis=1)
    return y
