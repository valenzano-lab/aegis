import numpy as np
from aegis.help import analyzer


def get_total_survivorship(container, iloc=-1):
    y = analyzer.get_total_survivorship(container)
    y = y.iloc[iloc]
    return y


def get_life_expectancy(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    pdf = phenotypes.iloc[:, :max_age]
    survivorship = pdf.cumprod(1)
    y = survivorship.sum(1)
    return y


def get_intrinsic_mortality(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    pdf = phenotypes.iloc[-1, :max_age]
    y = 1 - pdf
    return y


def get_intrinsic_survivorship(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    pdf = phenotypes.iloc[-1, :max_age]
    y = pdf.cumprod()
    return y


def get_fertility(container):
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    fertility = phenotypes.iloc[-1, max_age:]
    y = fertility
    return y


def get_cumulative_reproduction(container):
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13289
    # BUG fertility is 0 before maturity
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    survivorship = phenotypes.iloc[-1, :max_age].cumprod()
    fertility = phenotypes.iloc[-1, max_age:]
    y = (survivorship.values * fertility.values).cumsum()
    return y


def get_lifetime_reproduction(container):
    # BUG fertility is 0 before maturity
    phenotypes = container.get_df("phenotypes")
    max_age = container.get_config()["MAX_LIFESPAN"]
    survivorship = phenotypes.iloc[:, :max_age].cumprod(1)
    fertility = phenotypes.iloc[:, max_age:]
    y = np.sum((np.array(survivorship) * np.array(fertility)), axis=1)
    return y


def get_birth_structure(container):
    age_at_birth = container.get_df("age_at_birth")
    y = age_at_birth.iloc[-1]
    y /= y.sum()
    return y


def get_death_structure(container):
    age_at_genetic = container.get_df("age_at_genetic")
    age_at_overshoot = container.get_df("age_at_overshoot")

    t = -1
    pseudocount = 0
    y = (age_at_genetic.iloc[t] + pseudocount) / (
        age_at_overshoot.iloc[t] + age_at_genetic.iloc[t] + pseudocount
    )
    return y
