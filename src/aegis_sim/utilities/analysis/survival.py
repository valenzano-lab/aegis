# BASIC

# TODO add functions that convert mortality to survival to survivorship to counts, etc.


def get_mortality(survival):
    # TODO Ensure that you are slicing the phenotype array at right places
    y = 1 - survival
    return y


def get_survivorship(survival):
    # TODO Ensure that you are slicing the phenotype array at right places
    y = survival.cumprod(axis=1)
    return y


def get_total_mortality(interval_life_table):
    total_survivorship = get_total_survivorship(interval_life_table)
    total_mortality = -total_survivorship.pct_change(axis=1)
    return total_mortality


def get_total_survivorship(interval_life_table):
    # additive age structure records viability of each individual during its lifetime
    # when applied to a discrete cohort, additive age structure have the same shape as the survivorship curve
    # furthermore, if normalized, it is equivalent to the survivorship curve
    # when not applied to a cohort, the same holds if the population is stationary which approximately holds at short time scales
    total_survivorship = interval_life_table.div(interval_life_table.iloc[:, 0], axis=0)  # normalize
    return total_survivorship


# COMPLEX


def get_life_expectancy(phenotypes, AGE_LIMIT):
    # TODO Ensure that you are slicing the phenotype array at right places
    pdf = phenotypes.iloc[:, :AGE_LIMIT]
    survivorship = pdf.cumprod(1)
    y = survivorship.sum(1)
    return y


def get_longevity(survivorship, proportion_alive):
    """
    Longevity (omega) .. age at which proportion_alive of the population is still alive.
    Usually, this is set to 95% or 99%, sometimes called exceptional lifespan.
    """
    return survivorship.ge(proportion_alive).sum(1)
