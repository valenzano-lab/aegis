import numpy as np
from numba import njit
from aegis_sim import variables
from aegis_sim.utilities.funcs import profile_time


def recombination(genomes, RECOMBINATION_RATE):
    """Return recombined chromatids."""

    if RECOMBINATION_RATE == 0:
        return genomes

    # Recombine two chromatids but pass only one;
    #   thus double the number of chromatids, recobine,
    #   then return only one chromatid from each chromatid pair
    genomes = genomes[np.repeat(np.arange(len(genomes)), 2)]

    # Flatten loci and bits
    flat_genomes = genomes.reshape(len(genomes), 2, -1)

    # Get chromatids
    chromatid1 = flat_genomes[:, 0]
    chromatid2 = flat_genomes[:, 1]

    # Make choice array: when to take recombined and when to take original loci
    # -1 means synapse; +1 means clear
    rr = RECOMBINATION_RATE / 2  # / 2 because you are generating two random vectors (fwd and bkd)
    reco_fwd = (variables.rng.random(chromatid1.shape) < rr) * -2 + 1
    reco_bkd = (variables.rng.random(chromatid2.shape) < rr) * -2 + 1

    # Propagate synapse
    reco_fwd_cum = np.cumprod(reco_fwd, axis=1)
    reco_bkd_cum = np.cumprod(reco_bkd[:, ::-1], axis=1)[:, ::-1]

    # Recombine if both sites recombining
    reco_final = (reco_fwd_cum + reco_bkd_cum) == -2

    # Choose bits from first or second chromatid
    # recombined = np.empty(flat_genomes.shape, bool)
    recombined = np.empty(flat_genomes.shape, dtype=np.bool_)
    recombined[:, 0] = np.where(reco_final, chromatid2, chromatid1)
    recombined[:, 1] = np.where(reco_final, chromatid1, chromatid2)

    recombined = recombined.reshape(genomes.shape)
    recombined = recombined[::2]  # Look at first comment in the function

    return recombined


# # Loop version of the vectorized function above
# def recombination_via_pairs(genomes, RECOMBINATION_RATE):

#     if RECOMBINATION_RATE == 0:
#         return genomes

#     flat_genomes = genomes.reshape(len(genomes), 2, -1)

#     n_sites = flat_genomes.shape[-1]

#     n_recombination_sites = np.random.binomial(
#         n=n_sites,
#         p=RECOMBINATION_RATE,
#         size=len(flat_genomes),
#     )

#     # Produce all random numbers immediately
#     chiasmata_list = variables.rng.integers(
#         low=1,
#         high=n_sites,
#         size=(len(n_recombination_sites), max(n_recombination_sites)),
#         dtype=np.int32,
#     )  # [low, high)

#     for i, (chiasmata, n) in enumerate(zip(chiasmata_list, n_recombination_sites)):
#         for chiasma in chiasmata[:n]:
#             flat_genomes[i, 0, :chiasma], flat_genomes[i, 1, :chiasma] = (
#                 flat_genomes[i, 1, :chiasma],
#                 flat_genomes[i, 0, :chiasma],
#             )

#     unflattened_genomes = flat_genomes.reshape(genomes.shape)

#     return unflattened_genomes


@njit
def recombination_via_pairs_numba(flat_genomes, n_recombination_sites, chiasmata_list):
    for i in range(len(flat_genomes)):
        for j in range(n_recombination_sites[i]):
            chiasma = chiasmata_list[i, j]
            flat_genomes[i, 0, :chiasma], flat_genomes[i, 1, :chiasma] = (
                flat_genomes[i, 1, :chiasma].copy(),
                flat_genomes[i, 0, :chiasma].copy(),
            )
    return flat_genomes


def recombination_via_pairs(genomes, RECOMBINATION_RATE):
    if RECOMBINATION_RATE == 0:
        return genomes

    flat_genomes = genomes.reshape(len(genomes), 2, -1).copy()
    n_sites = flat_genomes.shape[-1]
    n_recombination_sites = np.random.binomial(n=n_sites, p=RECOMBINATION_RATE, size=len(flat_genomes))

    max_n = max(n_recombination_sites)
    chiasmata_list = variables.rng.integers(low=1, high=n_sites, size=(len(flat_genomes), max_n), dtype=np.int32)

    flat_genomes = recombination_via_pairs_numba(flat_genomes, n_recombination_sites, chiasmata_list)

    return flat_genomes.reshape(genomes.shape)
