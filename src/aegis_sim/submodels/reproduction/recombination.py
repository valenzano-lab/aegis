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
    """Optimized recombination using difference-array approach.

    Instead of copying genome slices for each chiasma (O(n_sites) per chiasma),
    we compute the net swap state per site using a difference array and prefix sum.
    Each chiasma at position c toggles sites [0, c), so we track toggle counts
    and only swap sites toggled an odd number of times.

    This avoids the expensive slice-copy pattern that scaled poorly with genome size.
    """
    n_individuals = len(flat_genomes)
    n_sites = flat_genomes.shape[2]

    for i in range(n_individuals):
        n_reco = n_recombination_sites[i]
        if n_reco == 0:
            continue

        # Build difference array for toggle counts
        counts = np.zeros(n_sites + 1, dtype=np.int32)
        for j in range(n_reco):
            c = chiasmata_list[i, j]
            counts[0] += 1
            if c < n_sites + 1:
                counts[c] -= 1

        # Prefix sum and swap where toggled odd number of times
        running = np.int32(0)
        for k in range(n_sites):
            running += counts[k]
            if running % 2 == 1:
                tmp = flat_genomes[i, 0, k]
                flat_genomes[i, 0, k] = flat_genomes[i, 1, k]
                flat_genomes[i, 1, k] = tmp

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


@njit
def _compute_swap_mask(swap_mask, n_recombination_sites, chiasmata_list):
    """Compute which bits should be swapped, using the same difference-array logic
    as recombination_via_pairs_numba.

    Instead of swapping in place, writes a boolean mask indicating which bit
    positions should be swapped between chromatids.
    """
    n = len(swap_mask)
    n_bits = swap_mask.shape[1]
    for i in range(n):
        n_reco = n_recombination_sites[i]
        if n_reco == 0:
            continue
        counts = np.zeros(n_bits + 1, dtype=np.int32)
        for j in range(n_reco):
            c = chiasmata_list[i, j]
            counts[0] += 1
            if c < n_bits + 1:
                counts[c] -= 1
        running = np.int32(0)
        for k in range(n_bits):
            running += counts[k]
            if running % 2 == 1:
                swap_mask[i, k] = 1


@njit
def recombination_via_pairs_packed_numba(packed_genomes, swap_mask_packed):
    """Apply crossover swaps on packed bytes using a packed swap mask.

    For each individual, for each byte position:
    - If swap_mask byte is 0x00: no swap (skip)
    - If swap_mask byte is 0xFF: swap entire bytes between chromatids
    - Otherwise: bitwise masking for partial-byte swaps
    """
    n = len(packed_genomes)
    n_packed_bytes = packed_genomes.shape[2]
    for i in range(n):
        for b in range(n_packed_bytes):
            mask = swap_mask_packed[i, b]
            if mask == 0:
                continue
            elif mask == 255:  # 0xFF
                tmp = packed_genomes[i, 0, b]
                packed_genomes[i, 0, b] = packed_genomes[i, 1, b]
                packed_genomes[i, 1, b] = tmp
            else:
                c0 = packed_genomes[i, 0, b]
                c1 = packed_genomes[i, 1, b]
                packed_genomes[i, 0, b] = (c0 & ~mask) | (c1 & mask)
                packed_genomes[i, 1, b] = (c1 & ~mask) | (c0 & mask)
    return packed_genomes


def recombination_via_pairs_packed(packed_genomes, n_loci, bits_per_locus, RECOMBINATION_RATE):
    """Recombination operating directly on packed uint8 arrays.

    Uses the same RNG draws as recombination_via_pairs (binomial + integers)
    to compute a swap mask, then applies swaps on packed bytes.

    Args:
        packed_genomes: np.uint8 array, shape (n_individuals, 2, n_packed_bytes)
        n_loci: number of loci
        bits_per_locus: bits per locus
        RECOMBINATION_RATE: crossover probability per site

    Returns:
        np.uint8 array, same shape, with crossovers applied
    """
    if RECOMBINATION_RATE == 0:
        return packed_genomes

    n = len(packed_genomes)
    n_total_bits = n_loci * bits_per_locus

    # Same RNG calls as recombination_via_pairs
    n_recombination_sites = np.random.binomial(n=n_total_bits, p=RECOMBINATION_RATE, size=n)

    max_n = max(n_recombination_sites)
    if max_n == 0:
        return packed_genomes

    chiasmata_list = variables.rng.integers(
        low=1, high=n_total_bits, size=(n, max_n), dtype=np.int32
    )

    # Compute swap mask using difference-array approach (same logic as numba kernel)
    swap_mask = np.zeros((n, n_total_bits), dtype=np.uint8)
    _compute_swap_mask(swap_mask, n_recombination_sites, chiasmata_list)

    # Pack the swap mask to uint8
    swap_mask_packed = np.packbits(swap_mask, axis=-1, bitorder='big')

    # Apply swaps on packed genomes
    packed_genomes = packed_genomes.copy()
    packed_genomes = recombination_via_pairs_packed_numba(packed_genomes, swap_mask_packed)
    return packed_genomes

