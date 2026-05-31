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


def get_fertility_intrinsic(container: Container, iloc=-1):
    ys = container.get_phenotype_intrinsic_interval("repr")
    max_iloc = ys.shape[0]
    ys = ys.iloc[iloc].pipe(lambda x: x).to_numpy()
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


def get_egg_number_after_reproduction(container: Container, iloc=None):
    ys = container.get_egg_number_after_reproduction().number
    max_iloc = None
    return ys, max_iloc


def get_resource_amount_before_scavenging(container: Container, iloc=None):
    ys = container.get_resource_amount_before_scavenging().resources
    max_iloc = None
    return ys, max_iloc


def get_resource_amount_after_scavenging(container: Container, iloc=None):
    ys = container.get_resource_amount_after_scavenging().resources
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


# ----- v3: lineage Muller plot + selection-coefficient trajectory -----------

def get_allele_freq_trajectory(container: Container, iloc=None):
    """Allele frequency over time at the injection locus.

    Returns (freq_series, None). Returns (None, None) if SelectionRecorder
    didn't write a log (ALLELE_INJECTION_STEP was 0 or LINEAGE_RATE was 0).
    """
    sel = container.get_selection_log()
    if sel is None or len(sel) == 0:
        return None, None
    return sel["allele_freq"].to_numpy(), None


def get_lineage_muller(container: Container, iloc=None):
    """Founder-lineage frequencies over time, as a (n_founders, n_steps) matrix.

    Returns (matrix_dict, None) where matrix_dict has keys "steps" and "stack".
    Returns (None, None) if LINEAGE_TRACING was off (no births.csv).
    """
    import numpy as np

    births = container.get_lineage_births()
    if births is None or len(births) == 0:
        return None, None
    deaths = container.get_lineage_deaths()

    parent_of = dict(zip(births["lineage_id"].tolist(), births["parent_lineage_id"].tolist()))
    birth_step = dict(zip(births["lineage_id"].tolist(), births["step"].tolist()))
    death_step = (
        dict(zip(deaths["lineage_id"].tolist(), deaths["step"].tolist()))
        if deaths is not None
        else {}
    )

    # Memoized founder lookup (climb parent chain until parent == -1)
    founder = {}

    def find_founder(lid):
        if lid in founder:
            return founder[lid]
        chain = []
        cur = lid
        while parent_of.get(cur, -1) != -1:
            if cur in founder:
                root = founder[cur]
                for x in chain:
                    founder[x] = root
                return root
            chain.append(cur)
            cur = parent_of[cur]
        for x in chain + [cur]:
            founder[x] = cur
        return cur

    for lid in parent_of:
        find_founder(lid)

    max_step = int(max(births["step"].max(), deaths["step"].max() if deaths is not None and len(deaths) else 0))
    steps = np.arange(0, max_step + 1)
    unique_founders = sorted(set(founder.values()))
    counts = {f: np.zeros(len(steps), dtype=np.int64) for f in unique_founders}

    for lid, b_step in birth_step.items():
        d_step = death_step.get(lid, max_step + 1)
        f = founder[lid]
        lo = int(b_step)
        hi = min(int(d_step), max_step + 1)
        if hi > lo:
            counts[f][lo:hi] += 1

    stack = np.stack([counts[f] for f in unique_founders], axis=0)
    return {"steps": steps, "stack": stack, "founders": unique_founders}, None
