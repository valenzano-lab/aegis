def get_sorted_allele_frequencies(interval_genotypes):
    total_frequency = interval_genotypes.sum(0)
    return interval_genotypes.T.assign(total=total_frequency).sort_values(by="total", ascending=False).T.iloc[:-1]


def get_derived_allele_freq(interval_genotypes):
    reference = interval_genotypes.round()
    derived_allele_freq = (
        interval_genotypes.iloc[1:].reset_index(drop=True) - reference.iloc[:-1].reset_index(drop=True)
    ).abs()
    return derived_allele_freq


def get_mean_allele_freq(interval_genotypes):
    mean_allele_freq = interval_genotypes.mean(0)
    return mean_allele_freq


def get_quantile_allele_freq(interval_genotypes, quantile):
    quantile_allele_freq = interval_genotypes.quantile(quantile)
    return quantile_allele_freq
