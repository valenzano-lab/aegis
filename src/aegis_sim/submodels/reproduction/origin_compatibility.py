import logging
import numpy as np
from aegis_sim import variables


def compute_position_independent_incompatibility(
    males,
    females,
    origins,
    ORIGIN_TRACKING,
    ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY,
):
    if (
        ORIGIN_TRACKING == "no_tracking"
        or ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY == 0
    ):
        return males, females

    male_origins = origins[males].reshape(len(males), -1)  # [n_pairs, flattened_size]
    female_origins = origins[females].reshape(len(females), -1)

    max_label = max(origins.max(), 0)

    male_props = (
        np.array([np.bincount(row, minlength=max_label + 1) for row in male_origins])
        / male_origins.shape[1]
    )
    female_props = (
        np.array([np.bincount(row, minlength=max_label + 1) for row in female_origins])
        / female_origins.shape[1]
    )

    # L1 distance per pair
    distances = np.abs(male_props - female_props).sum(axis=1)

    # Normalize L1 distances from [0, 2] to [0, 1]
    incompatibility_per_pair = distances / 2
    scaled_incompatibility_per_pair = (
        incompatibility_per_pair * ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY
    )

    random_probabilities = variables.rng.random(len(males))
    mask = random_probabilities < scaled_incompatibility_per_pair

    logging.debug(
        f"Proportion of pairs that failed to mate due to origin incompatibility: {mask.mean():.3f}"
    )

    return males[~mask], females[~mask]


def compute_position_dependent_incompatibility(
    males,
    females,
    origins,
    ORIGIN_TRACKING,
    ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY,
):
    if (
        ORIGIN_TRACKING == "no_tracking"
        or ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY == 0
    ):
        return males, females

    compatibility_per_pair = (
        np.equal(origins[males], (origins[females])).reshape(len(males), -1).mean(1)
    )
    incompatibility_per_pair = 1 - compatibility_per_pair
    scaled_incompatibility_per_pair = (
        incompatibility_per_pair * ORIGIN_INCOMPATIBILITY_REPRODUCTIVE_PENALTY
    )

    random_probabilities = variables.rng.random(len(males))
    mask = random_probabilities < scaled_incompatibility_per_pair

    logging.debug(
        f"Proportion of pairs that failed to mate due to origin incompatibility: {mask.mean():.3f}"
    )
    return males[~mask], females[~mask]
