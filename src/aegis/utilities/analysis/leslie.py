import numpy as np


def leslie_matrix(s, r):
    leslie = np.diag(s, k=-1)
    leslie[0] = r
    leslie[np.isnan(leslie)] = 0
    return leslie


def interval_leslie(interval_life_table, interval_birth_table, index):
    lt = interval_life_table[index]
    s = (1 + lt.pct_change())[1:]
    bt = interval_birth_table[index]
    r = (bt / lt).fillna(0)
    return leslie_matrix(s, r)


def intrinsic_leslie(intrinsic_mortality, intrinsic_fertility, index):
    """
    Hypothetical Leslie matrix that ignores extrinsic mortality.
    """
    m = intrinsic_mortality.iloc[index]
    s = 1 - m
    r = intrinsic_fertility.iloc[index]
    return leslie_matrix(s[:-1], r)


def leslie_breakdown(leslie):
    eigenvalues, eigenvectors = np.linalg.eig(leslie)
    dominant_index = np.argmax(np.abs(eigenvalues))
    dominant_eigenvector = eigenvectors[:, dominant_index]
    return {
        "growth_rate": np.max(np.abs(eigenvalues)),
        "stable_age_structure": dominant_eigenvector / dominant_eigenvector.sum(),
    }
