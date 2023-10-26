# analyze feather snapshots


# def get_total_mortality(cont):
#     total_survivorship = get_total_survivorship(cont)
    # total_mortality =


def get_total_survivorship(cont):
    # additive age structure records viability of each individual during its lifetime
    # when applied to a discrete cohort, additive age structure have the same shape as the survivorship curve
    # furthermore, if normalized, it is equivalent to the survivorship curve
    # when not applied to a cohort, the same holds if the population is stationary which approximately holds at short time scales
    additive_age_structure = cont.get_df("additive_age_structure")
    total_survivorship = additive_age_structure.div(
        additive_age_structure.iloc[:, 0], axis=0
    )  # normalize
    return total_survivorship


# analyze pickled populations

# analyze csvs
