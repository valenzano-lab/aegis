# analyze feather snapshots


def get_total_mortality(cont):
    return


def get_total_survivorship(cont):
    # cumulative ages records viability of each individual during its lifetime
    # when applied to a discrete cohort, cumulative ages have the same shape as the survivorship curve
    # furthermore, if normalized, it is equivalent to the survivorship curve
    # when not applied to a cohort, the same holds if the population is stationary which approximately holds at short time scales
    cumulative_ages = cont.get_df("cumulative_ages")
    total_survivorship = cumulative_ages.div(
        cumulative_ages.iloc[:, 0], axis=0
    )  # normalize
    return total_survivorship


# analyze pickled populations

# analyze csvs
